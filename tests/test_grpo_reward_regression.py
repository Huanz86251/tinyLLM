import unittest
from train.GRPO import compute_rewards_for_group

class FakeTokenizer:
    def encode(self,text,add_special_tokens=False):
        return text.split()

class GRPORewardRegressionTests(unittest.TestCase):
    def test_correct_boxed_cot_has_defined_format_and_length_scores(self):
        text='<|thought_start|> ' + ' '.join(f'step{i}' for i in range(60)) + '<|thought_end|> final \\boxed{42}'
        rewards,info=compute_rewards_for_group('#### 42',[text],FakeTokenizer())
        self.assertGreater(float(rewards[0]),1.0)
        self.assertTrue(info[0]['is_correct'])
        self.assertEqual(info[0]['format_score'],1.0)
        self.assertGreater(info[0]['length_score'],0.0)

    def test_degenerate_loop_is_penalized_even_with_correct_box(self):
        unit='repeat this bad phrase forever '
        text='<|thought_start|>'+unit*3+'<|thought_end|> \\boxed{42}'
        rewards,info=compute_rewards_for_group('#### 42',[text],FakeTokenizer())
        self.assertEqual(float(rewards[0]),-0.5)
        self.assertTrue(info[0]['loop_found'])

if __name__=='__main__':unittest.main(verbosity=2)

