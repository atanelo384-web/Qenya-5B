import os, sys, tempfile
os.environ.setdefault("QENYA_HOME", tempfile.mkdtemp(prefix="qenya_test_"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch


class TestVocab(unittest.TestCase):
    def test_corpus_and_bpe_roundtrip(self):
        from qenya import vocab as V
        texts = [V.generate_corpus_shard(i, 3000) for i in range(6)]
        bpe = V.QenyaBPE(vocab_size=2500).train(texts)
        s = "Hello world! def fib(n): return 42 Привет мир こんにちは"
        self.assertEqual(bpe.decode(bpe.encode(s)), s)
        sp = V.IM_START + "user x" + V.IM_END
        self.assertEqual(bpe.decode(bpe.encode(sp)), sp)

    def test_sandbox_build(self):
        from qenya import vocab as V
        paths = V.build_sandbox_corpus(total_tokens=4_000_000, shard_tokens=2_000_000)
        self.assertEqual(len(paths), 2)
        self.assertTrue(paths[0].exists())


class TestModel(unittest.TestCase):
    def test_forward_backward_and_cache(self):
        from qenya.model import QenyaLLM, QenyaConfig, QENYA_TINY, count_params
        cfg = QenyaConfig(**QENYA_TINY); cfg.vocab_size = 900
        m = QenyaLLM(cfg)
        ids = torch.randint(10, 900, (1, 24))
        lab = ids.clone(); lab[:, :6] = -100
        logits, loss = m(ids, lab)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        cache = m.new_cache()
        m(ids[:, :8], kv_cache=cache)
        for c in cache: c["pos"] = 8
        for i in range(8, 24):
            o2, _ = m(ids[:, i:i+1], kv_cache=cache)
            for c in cache: c["pos"] += 1
        full, _ = m(ids)
        self.assertLess((o2[0, -1] - full[0, -1]).abs().max().item(), 1e-3)

    def test_5b_config_size(self):
        from qenya.model import QenyaConfig, QENYA_5B, count_params
        n = count_params(QenyaConfig(**QENYA_5B))
        self.assertGreater(n, 4.5e9)   # действительно ~5B
        self.assertLess(n, 6.0e9)


class TestVision(unittest.TestCase):
    def test_vit_shape(self):
        from qenya.vision import QenyaVision
        v = QenyaVision(llm_dim=512, image_tokens=64)
        out = v.encode(torch.rand(1, 3, 128, 128))
        self.assertEqual(tuple(out.shape), (1, 64, 512))


class TestTools(unittest.TestCase):
    def test_write_run_tests(self):
        from qenya.agent import SandboxTools, parse_tool_call
        t = SandboxTools()
        code = ("import unittest\n"
                "def score(xs): return sum(xs)\n"
                "class T(unittest.TestCase):\n"
                "    def test_s(self): self.assertEqual(score([1,2,3]), 6)\n")
        self.assertIn("OK", t.write_file("demo/test_game.py", code))
        self.assertIn("PASS", t.run_tests())
        self.assertIn("45", t.run_python("print(sum(range(10)))"))
        c = parse_tool_call('x {"tool": "web_fetch", "args": {"url": "http://a.b"}} y')
        self.assertEqual(c["tool"], "web_fetch")

    def test_sandbox_escape_blocked(self):
        from qenya.agent import SandboxTools
        t = SandboxTools()
        with self.assertRaises(PermissionError):
            t.write_file("../../etc/passwd", "hack")


if __name__ == "__main__":
    unittest.main()
