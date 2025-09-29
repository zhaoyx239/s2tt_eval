import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf


client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为:api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


base64_audio = encode_audio("a.wav")
input_100 = """
请根据以下标准对语音翻译结果进行评分(总分 0-100 分):

1. 信息完整性(20分):是否完整传达了音频中的所有关键信息(如时间、地点、事件、逻辑关系等)。
    - 完整:20分
    - 遗漏1个次要信息:15-17分
    - 遗漏1个关键信息:12-14分
    - 遗漏2个以上关键信息:6-11分
    - 几乎无信息:0-5分
2. 语言准确性(20分):是否使用正确的词汇、语法、时态、拼写、标点。
    - 完全正确:20分
    - 1-2处轻微错误:15-17分
    - 2-3处错误:12-14分
    - 多处错误:6-11分
    - 大量错误:0-5分
3. 表达自然度(20分):是否符合目标语言的自然表达习惯，是否生硬或中式。
    - 完全自然:20分
    - 1-2处略显生硬:15-17分
    - 2-3处不自然:12-14分
    - 多处不自然:6-11分
    - 严重不自然:0-5分
4. 忠实度(20分):是否忠实于原文，有无添加、遗漏或曲解。
    - 完全忠实:20分
    - 1处轻微偏差:15-17分
    - 1-2处偏差:12-14分
    - 多处偏差:6-11分
    - 严重偏离:0-5分
5. 语法与结构完整性(20分):是否构成完整句子，语法结构是否正确。
    - 完整句子:20分
    - 有轻微语法问题:15-17分
    - 有1-2处语法错误:12-14分
    - 语法混乱:6-11分
    - 完全无法构成句子:0-5分

请按以上五项对每个翻译分别打分，然后相加得出总分(0-100)。输出格式为给几个翻译版本就输出几行，例如A 98为一行表示翻译A得到98分。
A Since the end of June, when Hohhot took the lead in announcing the cancellation of its home purchase restrictions,
B Since late June, after Hohhot first announced to cancel the property purchase limits,
C From June end, Hohhot first said to remove property limits,
"""
input_5 = """
请根据翻译结果的整体质量，参考以下描述给出一个整体的星级评价：

​​★☆☆☆☆ （1星） - 差​​
基本无法理解，信息严重缺失或完全错误。
语言支离破碎，有大量致命错误。
完全无法达到沟通目的。
​​★★☆☆☆ （2星） - 较差​​
仅能捕捉到极少量关键词，但核心信息缺失或错误。
语言不连贯，存在大量语法和用词错误，理解非常困难。
无法有效传递主要信息。
​​★★★☆☆ （3星） - 一般​​
能够传达部分核心信息，但存在关键信息遗漏或明显错误。
语言表达不流畅，有较多不自然的直译或语法错误，但勉强可以理解大意。
需要听者花费较多精力去猜测和推断。
​​★★★★☆ （4星） - 良好​​
准确传达了绝大部分关键信息，可能有个别次要信息遗漏。
语言基本准确、通顺，表达整体自然，虽有极少数瑕疵但不影响理解。
能够有效实现沟通目的。
​​★★★★★ （5星） - 优秀​​
完整、准确地传达了原文的所有关键信息和细节。
语言地道、流畅，完全符合目标语言的表达习惯，无语法错误。
阅读起来如同母语者自然表达，无需任何猜测。
​​评分方式：​​

根据上述整体描述，为下列每个翻译结果直接评定一个最终的星级（1星至5星）。
A Since the end of June, when Hohhot took the lead in announcing the cancellation of its home purchase restrictions,
C From June end, Hohhot first said to remove property limits,
"""
completion = client.chat.completions.create(
    model="qwen3-omni-flash",  # 模型为Qwen3-Omni-Flash时，请关闭思考模式，否则代码会报错
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{base64_audio}",
                        "format": "mp3",
                    },
                },
                {"type": "text", "text": input_100},
            ],
        },
    ],
    # 设置输出数据的模态，当前支持两种:["text","audio"]、["text"]
    modalities=["text"],
    audio={"voice": "Cherry", "format": "wav"},
    # stream 必须设置为 True，否则会报错
    stream=True,
    stream_options={"include_usage": True},
)
print("模型回复：")
audio_base64_string = ""
for chunk in completion:
    # 处理文本部分
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

    # 收集音频部分
    if (
        chunk.choices
        and hasattr(chunk.choices[0].delta, "audio")
        and chunk.choices[0].delta.audio
    ):
        audio_base64_string += chunk.choices[0].delta.audio.get("data", "")

    # 4. 保存音频文件
if audio_base64_string:
    wav_bytes = base64.b64decode(audio_base64_string)
    audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
    sf.write("audio_assistant.wav", audio_np, samplerate=24000)
    print("\n音频文件已保存至：audio_assistant.wav")

# for chunk in completion:
#     if chunk.choices:
#         print(chunk.choices[0].delta)
#     else:
#         print(chunk.usage)
