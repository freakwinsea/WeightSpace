"""Free: which tokens does each candidate neuron promote, given the sign it fires with before Wait?"""
import torch, ws
if __name__ == "__main__":
    s = ws.Session("Qwen/Qwen3.5-2B", budgeted=False)
    m, tok = s.model, s.tok
    W = (m.lm_head.weight.float() * m.model.norm.weight.float())
    for li, j, sign in [(16, 2431, -1), (17, 873, -1), (18, 3368, -1), (20, 3748, -1), (22, 5041, -1), (15, 1958, 1), (18, 3801, -1)]:
        col = m.model.layers[li].mlp.down_proj.weight[:, j].float() * sign
        lg = W @ col
        top = torch.topk(lg, 12).indices.tolist()
        bot = torch.topk(-lg, 8).indices.tolist()
        print(f"L{li}.{j} (fires {'+' if sign>0 else '-'}): promotes {[tok.decode([i]) for i in top]} | suppresses {[tok.decode([i]) for i in bot]}")
