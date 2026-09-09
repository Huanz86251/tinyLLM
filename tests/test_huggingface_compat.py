import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from model.config import Config
from model.model import TinyLLM
from tools.export_huggingface import export_model


def tiny_config() -> Config:
    return Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        train_maxlength=64,
        mlp_ratio_front=2.0,
        mlp_ratio_mid=2.0,
        mlp_ratio_back=2.0,
        mlp_mid_start=1,
        mlp_back_start=2,
        mlp_ratio_overrides={},
        use_adaptive_softmax=False,
        rope_type="default",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


class HuggingFaceCompatibilityTest(unittest.TestCase):
    def test_dynamic_auto_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            exported = root / "exported"
            checkpoint.mkdir()
            model = TinyLLM(tiny_config()).eval()
            model.save_pretrained(checkpoint, safe_serialization=True)
            (checkpoint / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            export_model(Namespace(
                checkpoint=checkpoint,
                tokenizer=checkpoint,
                output=exported,
                base_model=None,
                vision_tower=None,
                normalize_lora_wrappers=False,
            ))

            loaded = AutoModelForCausalLM.from_pretrained(
                exported,
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
            ids = torch.tensor([[1, 3, 4]])
            mask = torch.ones_like(ids)
            expected = model(input_ids=ids, attention_mask=mask)["logits"]
            actual = loaded(input_ids=ids, attention_mask=mask)["logits"]
            self.assertTrue(torch.equal(expected, actual))
            generated = loaded.generate(ids, max_new_tokens=2, do_sample=False)
            self.assertEqual(tuple(generated.shape), (1, 5))

            config = json.loads((exported / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                config["auto_map"]["AutoModelForCausalLM"],
                "modeling_tinyllm.TinyLLM",
            )

    def test_native_lora_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyLLM(tiny_config())
            model.attach_lora_adapter("demo", rank=4, alpha=8.0, dropout=0.0, target="all")
            for key, value in model.state_dict().items():
                if ".adapters.demo." in key:
                    value.copy_(torch.randn_like(value))
            model.save_lora_pretrained(
                tmp,
                "demo",
                rank=4,
                alpha=8.0,
                dropout=0.0,
                target="all",
                base_model_name_or_path="owner/tinyllm-base",
            )

            fresh = TinyLLM(tiny_config())
            metadata = fresh.load_lora_pretrained(tmp)
            self.assertEqual(metadata["adapter_name"], "demo")
            self.assertGreater(metadata["tensor_count"], 0)
            expected = model.get_lora_state_dict("demo")
            actual = fresh.get_lora_state_dict("demo")
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)


if __name__ == "__main__":
    unittest.main()
