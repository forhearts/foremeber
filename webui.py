"""ForeMeber WebUI 演示（Gradio）

用法:
    python webui.py            # 需本地 14B 服务(8081)
    python webui.py --offline  # 离线演示（不调模型，只测记忆+拼装）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from foremeber import ForeMeber
from remember.character import load_all_characters
from remember.memory import MemorySystem

FIXTURES = Path(__file__).resolve().parent / "tests" / "fixtures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--offline", action="store_true", help="离线演示（不调模型）")
    args = ap.parse_args()

    fm = ForeMeber(memory=MemorySystem())
    fm.chars = load_all_characters(FIXTURES / "characters")
    if not fm.chars:
        fm.chars = {c.id: c for c in _demo_chars()}

    session = {"cid": "aila"}

    def set_character(cid):
        session["cid"] = cid
        char = fm.chars[cid]
        return f"**{char.name}**（{char.identity}）— {char.personality}"

    def chat(player_text, history):
        if not player_text.strip():
            return "", history
        cid = session["cid"]
        char = fm.chars[cid]
        if args.offline:
            # 离线演示：只展示拼装结果
            p = fm.build_prompt(cid, player_text, "集市摊位")
            reply = f"[离线演示]\n\nsystem:\n{p['system']}\n\nuser:\n{p['user']}"
        else:
            reply = fm.chat(cid, player_text, "集市摊位")
        history = history or []
        history.append({"role": "user", "content": player_text})
        history.append({"role": "assistant", "content": reply})
        return "", history

    char_choices = [(c.name, cid) for cid, c in fm.chars.items()]

    with gr.Blocks(title="ForeMeber") as demo:
        gr.Markdown("# 🎭 ForeMeber — 记忆驱动的 NPC 对话")
        with gr.Row():
            with gr.Column(scale=1):
                char_dd = gr.Dropdown(char_choices, value="aila", label="角色")
                char_info = gr.Markdown()
            with gr.Column(scale=2):
                chatbox = gr.Chatbot(label="对话")
                msg = gr.Textbox(label="你说", lines=2)
                btn = gr.Button("发送", variant="primary")

        char_dd.change(set_character, inputs=char_dd, outputs=char_info)
        btn.click(chat, inputs=[msg, chatbox], outputs=[msg, chatbox])
        msg.submit(chat, inputs=[msg, chatbox], outputs=[msg, chatbox])
        set_character("aila")

    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)


def _demo_chars():
    """兜底演示角色（无 fixtures 时）"""
    from remember.character import Character
    return [
        Character(name="艾拉", id="aila", identity="流浪商人",
                  personality="警惕、爱钱、嘴硬心软", speech_style="短句、带刺",
                  goal="赚钱，寻找失踪的妹妹",
                  background="出生于边境村庄，战乱后四处流浪经商，妹妹被掳走。",
                  greetings=["一个路过的商人。别靠太近。"]),
    ]


if __name__ == "__main__":
    main()
