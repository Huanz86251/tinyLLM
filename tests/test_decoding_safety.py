import json, unittest, time
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import torch
from inference.decoding import SafeDecoder, text_loop, trim_history, completed_boxed_answer, low_information_tail

class Tokenizer:
    eos_token_id=5
    def convert_tokens_to_ids(self,s): return 5
    def encode(self,s,**kwargs): return [5,5]
    def decode(self,ids,**kwargs):
        names={0:"apples",1:"oranges",2:"1",3:"+",4:"trees",5:"<eos>"}
        return "".join(names.get(i,"word") for i in ids)

class SafetyTests(unittest.TestCase):
    def decoder(self,mode="math"):
        d=SafeDecoder(Tokenizer(),torch.device("cpu"),mode)
        d.config.update(strategy="greedy",repetition_penalty=1.2,frequency_penalty=0,no_repeat_ngram_size=0)
        return d
    def test_positive_penalty(self):
        d=self.decoder();token,_=d.choose(torch.tensor([[10.,9.2,-20,-20,-20,-20]]),[0]);self.assertEqual(token,1)
    def test_negative_penalty(self):
        d=self.decoder();token,_=d.choose(torch.tensor([[-1.,-1.1,-20,-20,-20,-20]]),[0]);self.assertEqual(token,1)
    def test_math_digits_preserved(self):
        d=self.decoder();token,_=d.choose(torch.tensor([[1.,1.,2.,0.,0.,0.]]),[2]*40);self.assertEqual(token,2)
    def test_eos_preserved(self):
        d=self.decoder();token,_=d.choose(torch.tensor([[.9,0,0,0,0,1.]]),[5]*40);self.assertEqual(token,5)
    def test_nan_is_stopped(self):
        d=self.decoder();token,problem=d.choose(torch.tensor([[float("nan"),1,0,0,0,0]]),[]);self.assertIsNone(token);self.assertEqual(problem["kind"],"invalid_logits")
    def test_infinity_is_stopped(self):
        d=self.decoder();self.assertIsNone(d.choose(torch.full((1,6),-float("inf")),[])[0])
    def test_actual_ted_loop(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "vision_repetition.json"
        if not fixture.is_file():
            self.skipTest("optional captured repetition fixture is not distributed")
        data=json.loads(fixture.read_text(encoding="utf-8"))
        cleaned,found=trim_history(data["text"])
        self.assertIsNotNone(found)
        self.assertLess(len(cleaned),len(data["text"])//2)
        d=SafeDecoder(Tokenizer(),torch.device("cpu"),"vision")
        self.assertIsNotNone(d.inspect(data["text"],list(range(100))))
    def test_normal_math_repetition_is_not_cycle(self):
        text="Step 1: x^2 + 1 = y.\nStep 2: x^2 = y - 1.\nStep 3: x = sqrt(y - 1).\nThe equation x^2 = y - 1 is equivalent."
        self.assertIsNone(text_loop(text))
        self.assertIsNone(text_loop(("x^2+1=y; "*12)))
    def test_ngram_blocks_prose(self):
        d=self.decoder("chat");d.config.update(no_repeat_ngram_size=3)
        token,_=d.choose(torch.tensor([[0.,0.,-20,-20,10.,-10.]]),[0,1,4,0,1])
        self.assertNotEqual(token,4);self.assertGreater(d.stats["ngram_tokens_blocked"],0)
    def test_math_ngram_does_not_block_number(self):
        d=self.decoder();d.config.update(no_repeat_ngram_size=3)
        token,_=d.choose(torch.tensor([[0.,0.,10.,-20,0.,-10.]]),[0,1,2,0,1]);self.assertEqual(token,2)
    def test_seed_reproducible(self):
        a=SafeDecoder(Tokenizer(),torch.device("cpu"),"vision");b=SafeDecoder(Tokenizer(),torch.device("cpu"),"vision")
        scores=torch.tensor([[1.,1.1,.5,.4,1.,-4.]])
        self.assertEqual([a.choose(scores,[])[0] for _ in range(12)],[b.choose(scores,[])[0] for _ in range(12)])

    def test_box_waits_for_closing_math_delimiter(self):
        self.assertFalse(completed_boxed_answer(r"The answer is $\boxed{7}"))
        self.assertTrue(completed_boxed_answer(r"The answer is $\boxed{7}$"))
        self.assertFalse(completed_boxed_answer(r"\(\boxed{7}"))
        self.assertTrue(completed_boxed_answer(r"\(\boxed{7}\)"))
    def test_box_handles_nested_fraction_and_thought(self):
        self.assertFalse(completed_boxed_answer(r"\boxed{\frac{1}{2}"))
        self.assertTrue(completed_boxed_answer(r"\boxed{\frac{1}{2}}"))
        self.assertFalse(completed_boxed_answer(r"<|thought_start|>try \boxed{2}"))
        self.assertTrue(completed_boxed_answer(r"<|thought_start|>try 2<|thought_end|>\boxed{2}"))

    def run_scripted_engine(self, *, invalid=False, timeout=False):
        from inference.chat_engine import ChatEngine
        phrase='图片中的主题包含蓝色区域和红色区域。'
        chars=list(dict.fromkeys(phrase))
        ids=[chars.index(c) for c in phrase]
        class CharTokenizer:
            eos_token_id=len(chars)
            pad_token_id=None
            def convert_tokens_to_ids(self,name):return self.eos_token_id
            def encode(self,text,**kwargs):return [self.eos_token_id,self.eos_token_id]
            def decode(self,values,**kwargs):return ''.join(chars[i] for i in values if i<len(chars))
        class ScriptedModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
                self.cfg=SimpleNamespace();self.index=0
            def forward(self,**kwargs):
                scores=torch.full((1,1,len(chars)+1), -100.)
                if invalid:scores[:]=float('nan')
                else:scores[0,0,ids[self.index%len(ids)]]=100.
                self.index+=1
                return {'logits':scores,'past_states':None}
        def configured(tokenizer,device,mode):
            d=SafeDecoder(tokenizer,device,mode)
            # Isolate the stop guard from preventive sampling/ngram controls.
            d.config.update(strategy='greedy',no_repeat_ngram_size=0)
            if timeout:d.config['max_generation_seconds']=-1
            return d
        engine=ChatEngine();engine.model=ScriptedModel();engine.tokenizer=CharTokenizer()
        events=[]
        with patch('inference.chat_engine.SafeDecoder',configured):
            result=engine._decode([0],None,'vision',2048,time.perf_counter(),events.append)
        return result,events
    def test_stream_loop_is_stopped_and_raw_output_retained(self):
        result,events=self.run_scripted_engine()
        self.assertEqual(result['stop_reason'],'repetition')
        self.assertLess(result['tokens'],200)
        self.assertLess(len(result['text']),len(result['raw_text']))
        self.assertTrue(any(e['type']=='guard' for e in events))
    def test_stream_invalid_logits_stops_without_invented_text(self):
        result,_=self.run_scripted_engine(invalid=True)
        self.assertEqual(result['stop_reason'],'invalid_logits')
        self.assertEqual(result['text'],'')
    def test_stream_time_limit_stops(self):
        result,_=self.run_scripted_engine(timeout=True)
        self.assertEqual(result['stop_reason'],'time_limit')
        self.assertEqual(result['tokens'],0)

    def test_min_p_relative_floor_filters_tail(self):
        d=self.decoder("chat")
        d.config.update(strategy="sample",temperature=1,top_k=6,top_p=1,min_p=0.2)
        token,problem=d.choose(torch.tensor([[10.,7.,6.,5.,4.,-5.]]),[])
        self.assertIsNone(problem)
        self.assertEqual(token,0)
        self.assertGreaterEqual(d.stats["min_p_tokens_filtered"],4)
    def test_min_p_disabled_for_deterministic_modes(self):
        self.assertEqual(SafeDecoder(Tokenizer(),torch.device("cpu"),"math").config["min_p"],0.0)
        self.assertEqual(SafeDecoder(Tokenizer(),torch.device("cpu"),"arc").config["min_p"],0.0)

    def test_vision_numeric_punctuation_tail_is_stopped(self):
        text='图例中指出其他区域标记为“2010-12-11-10-11-11-12-11-11-12.......... ......... (100% 11-12) ........ (20% 11-11-12-.........)'
        found=low_information_tail(text,64)
        self.assertIsNotNone(found)
        self.assertEqual(found['kind'],'low_information_tail')
        self.assertLess(found['keep_chars'],len(text))

    def test_low_information_tail_trims_dangling_clause(self):
        text='这是一句完整说明。图例还写着2010-12-11-10-11-12.......... ......... (100% 11-12) ........ (20% 11-12)'
        found=low_information_tail(text,64)
        self.assertEqual(text[:found['keep_chars']],'这是一句完整说明。')

    def test_normal_prose_with_year_and_percent_is_not_tail(self):
        text='图表显示，2010 年的比例为 20%，随后在 2011 年回落。这个结论来自图中的两条曲线。'
        self.assertIsNone(low_information_tail(text,64))
if __name__=="__main__":unittest.main(verbosity=2)

