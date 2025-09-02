import random
import os
import json
from tokenizers import models,pre_tokenizers,decoders,trainers,Tokenizer,normalizers
import regex as re
from transformers import PreTrainedTokenizerFast,AutoTokenizer

from tqdm import tqdm
import tempfile
random.seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = "16"
def count_jsonl(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                n += 1
            except json.JSONDecodeError:

                pass
    return n

def readjsonl(filename,json_length,sample_ratio=0.3,tail_size=100):
    with open(filename, encoding="utf-8") as f:

        ratio_lines = int(json_length * sample_ratio)+tail_size
        tail_start = json_length - tail_size
        i=0
        for i,line in enumerate(tqdm(f, total=ratio_lines, unit="lines", desc="Reading JSONL")):
            if i < tail_start:
                if random.random() > sample_ratio:
                    continue

            yield json.loads(line)["text"]

def build_tokenizer(tokenizer_dir, dataset_path):



    total_lines = count_jsonl(dataset_path)
    print(f"[info] total lines ≈ {total_lines:,}")

    extra_symbols = "(){}[]<>:;.,。，‘’“”！=+-*/%<>&|^!~#@'\"`\\_?$：；？《》——（）【】、　·•‧※℃°‰‱〃々≠≈≥≤±×÷√∞∑∏∂∇∈∩∪⊂⊆⊃⊇∧∨¬→←↑↓↔⇒⇔¥€£▪▫□…✓¡¿ªº¨¯´¸¬¦–—˜-‒―‖‛‟‱′″‴‵‶‷‸⁃⁅⁇⁈⁉‽⁃⁅⁆⁇⁈⁉⁌⁍⁎⁏⁑⁒⁓⁖⁗⁚¢₡₤₫∀∁∃∄∅∆∫∬∭∮∯∰∱∲∳∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≉≊≋≌⊗⊕∐−∓∔∕∖∗∘∙∛∜∝∟∠∡∢∣∤∥∦⋂⋃πταβγδεζηθικλμνξορσςυφχψωℕℤℚℝℂ↕⇑⇓⇐↦↪↩™®©ℓ№Ωᴂɛᴈɝʊᴧ"

    initial_alphabet = list(dict.fromkeys(list(pre_tokenizers.ByteLevel.alphabet()) + list(extra_symbols)))

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    PUNCT_OR_SYMBOL = r"[^\p{L}\p{N}\s_]"
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement="▁",prepend_scheme="never"),
        pre_tokenizers.Split(pattern=PUNCT_OR_SYMBOL, behavior="isolated"),
    ])
    tokenizer.decoder = decoders.Metaspace(replacement="▁",prepend_scheme="never")
    reserved = [f"<|reserved_{i}|>" for i in range(10)]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|unk|>","<|im_start|>system","<|im_start|>user","<|im_start|>assistant"] + reserved
    trainer=trainers.UnigramTrainer(
        vocab_size=20000,
        show_progress=True,
        initial_alphabet =initial_alphabet,
        special_tokens=specials,
        unk_token="<|unk|>",
        shrinking_factor = 0.7,
        n_sub_iterations=2
    )

    tokenizer.train_from_iterator(readjsonl(dataset_path,total_lines), trainer,length=total_lines)
    tok=PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|unk|>",
    )

    tok.clean_up_tokenization_spaces = False

    tok.chat_template=    ("{% if messages and messages[0]['role'] == 'system' %}"
    "{{ '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{% set start = 1 %}"
    "{% else %}"
    "{{ '<|im_start|>system\\n你是一个乐于助人的中文助手<|im_end|>\\n' }}"
    "{% set start = 0 %}"
    "{% endif %}"

    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"

    "{% for message in messages[start:] %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"

    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    tok.model_max_length=16384
    tok.padding_side = "left"

    tok.save_pretrained(tokenizer_dir)

def verify(tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(tokenizer)
    message=[
        {"role":"user","content":"Hello"},
        {"role":"assistant","content":"世界"},
        {"role": "user", "content": "for i in range(10): print(i==None)  # Python代码"},
        {"role": "assistant", "content": "你好，print(x==None)?函数‱   def foo(x:int)->List[str]: return x.split('_')"},

    ]
    nfkc = normalizers.NFKC()
    makeup_message=tokenizer.apply_chat_template(message,tokenize= False,add_generation_prompt=True)
    makeup_message=nfkc.normalize_str(makeup_message)
    print("apply template:",makeup_message)
    id_message=tokenizer.encode(makeup_message,add_special_tokens=False)
    print("id message:",id_message[:10])

    decode_message=tokenizer.decode(id_message,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    decode_message=nfkc.normalize_str(decode_message)
    print("decode:",decode_message)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_dir = os.path.join(BASE_DIR, "data", "wudao_filtered_5gb.jsonl")
    tokenizer_dir = os.path.join(BASE_DIR, "model")
    build_tokenizer(tokenizer_dir,dataset_dir)
    verify(tokenizer_dir)

if __name__=="__main__":
    main()