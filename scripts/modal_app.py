"""Modal 部署：7B 角色扮演 API（备选方案）

用法:
    pip install modal
    modal token new   # 浏览器登录
    modal deploy scripts/modal_app.py
    输出中会有 https://xxx.modal.app 地址
"""
import modal

app = modal.App("npc-roleplay")

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "accelerate", "gradio"
).run_commands("apt-get update && apt-get install -y cloudflared")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 角色库
CHARACTERS = {
    "aila": {"name": "艾拉", "identity": "流浪商人", "personality": "警惕、爱钱、嘴硬心软", "speech_style": "短句、带刺"},
    "bruno": {"name": "布鲁诺", "identity": "酒馆老板", "personality": "豪爽、爱吹牛", "speech_style": "粗犷、夸张"},
    "kara": {"name": "卡拉", "identity": "铁匠", "personality": "沉默寡言", "speech_style": "话少直接"},
    "orin": {"name": "奥林", "identity": "年轻守卫", "personality": "认真天真", "speech_style": "规矩热情"},
    "morgan": {"name": "摩根", "identity": "老猎人", "personality": "经验丰富", "speech_style": "简短直接"},
    "luna": {"name": "露娜", "identity": "女巫学徒", "personality": "好奇心重", "speech_style": "活泼跳跃"},
    "victor": {"name": "维克托", "identity": "反派手下", "personality": "傲慢嘴毒", "speech_style": "刻薄"},
    "elda": {"name": "艾尔达", "identity": "村长夫人", "personality": "慈祥爱八卦", "speech_style": "温和唠叨"},
}


@app.cls(image=image, gpu="A10G", timeout=600, allow_concurrent_inputs=16)
class Roleplay:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.model.eval()

    @modal.method()
    def generate(self, cid: str, player_input: str, scene: str, affection: int) -> str:
        import torch
        char = CHARACTERS.get(cid, CHARACTERS["aila"])
        sys_p = (
            f"你是游戏NPC「{char['name']}」，{char['identity']}。性格：{char['personality']}。"
            f"说话风格：{char['speech_style']}。请保持角色，回复要短（≤40字），"
            f"绝不承认自己是AI。只输出台词，禁止旁白。"
        )
        user_p = f"[当前状态]\n场景：{scene}\n好感度：{affection}\n\n[玩家]\n{player_input}"
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64, temperature=0.7,
                                      top_p=0.9, do_sample=True,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@app.function(image=image, gpu="A10G", allow_concurrent_inputs=16)
@modal.web_endpoint(method="POST")
def chat(body: dict):
    rp = Roleplay()
    reply = rp.generate.remote(body["cid"], body["player_input"], body.get("scene", "集市"), body.get("affection", 0))
    return {"reply": reply}
