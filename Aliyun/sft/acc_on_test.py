from openai import AsyncOpenAI
import requests
import json
import asyncio
import os

# 若没有配置环境变量，请用EAS服务的Token将下行替换为：token = 'YTA1NTEzMzY3ZTY4Z******************'
token = os.environ.get("Token")
# <调用地址>后面有 “/v1”不要去除
client = OpenAI(
    api_key=token,
    base_url=f'调用地址/v1',
)

if token is None:
    print("请配置环境变量Token, 或直接将Token赋值给token变量")
    exit()

system_prompt = """你是一个专业的信息抽取助手，专门负责从中文文本中提取收件人的结构化信息。

## 任务说明
请根据给定的输入文本，准确提取并生成包含以下六个字段的JSON格式输出：
- province: 省份/直辖市/自治区（必须是完整的官方名称，如"河南省"、"上海市"、"新疆维吾尔自治区"等）
- city: 城市名称（包含"市"字，如"郑州市"、"西安市"等）
- district: 区县名称（包含"区"、"县"等，如"金水区"、"雁塔区"等）
- specific_location: 具体地址（街道、门牌号、小区、楼栋等详细信息）
- name: 收件人姓名（完整的中文姓名）
- phone: 联系电话（完整的电话号码，包括区号）

## 抽取规则
1. **地址信息处理**：
   - 必须准确识别省、市、区的层级关系
   - 省份名称必须使用官方全称（如"河南省"而非"河南"）
   - 直辖市的province和city字段应该相同（如都填"上海市"）
   - specific_location应包含详细的街道地址、小区名称、楼栋号等

2. **姓名识别**：
   - 准确提取完整的中文姓名，包括复姓
   - 包括少数民族姓名

3. **电话号码处理**：
   - 提取完整的电话号码，保持原有格式

## 输出格式
请严格按照以下JSON格式输出，不要添加任何解释性文字：
{
  "province": "省份名称",
  "city": "城市名称", 
  "district": "区县名称",
  "specific_location": "详细地址",
  "name": "收件人姓名",
  "phone": "联系电话"
}"""


def compare_address_info(actual_address_str, predicted_address_str):
    """比较两个JSON字符串表示的地址信息是否相同"""
    try:
        # 解析实际地址信息
        if actual_address_str:
            actual_address_json = json.loads(actual_address_str)
        else:
            actual_address_json = {}

        # 解析预测地址信息
        if predicted_address_str:
            predicted_address_json = json.loads(predicted_address_str)
        else:
            predicted_address_json = {}

        # 直接比较两个JSON对象是否完全相同
        is_same = actual_address_json == predicted_address_json

        return {
            "is_same": is_same,
            "actual_address_parsed": actual_address_json,
            "predicted_address_parsed": predicted_address_json,
            "comparison_error": None
        }

    except json.JSONDecodeError as e:
        return {
            "is_same": False,
            "actual_address_parsed": None,
            "predicted_address_parsed": None,
            "comparison_error": f"JSON解析错误: {str(e)}"
        }
    except Exception as e:
        return {
            "is_same": False,
            "actual_address_parsed": None,
            "predicted_address_parsed": None,
            "comparison_error": f"比较错误: {str(e)}"
        }


async def predict_single_conversation(conversation_data):
    """预测单个对话的标签"""
    try:
        # 提取user content（去除assistant message）
        messages = conversation_data.get("messages", [])
        user_content = None

        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content", "")
                break

        if not user_content:
            return {"error": "未找到用户消息"}

        response = await client.chat.completions.create(
            model="Qwen3-0.6B",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            extra_body={
                "enable_thinking": False
            }
        )

        predicted_labels = response.choices[0].message.content.strip()
        return {"prediction": predicted_labels}

    except Exception as e:
        return {"error": f"预测失败: {str(e)}"}


async def process_batch(batch_data, batch_id):
    """处理一批数据"""
    print(f"处理批次 {batch_id}，包含 {len(batch_data)} 条数据...")

    tasks = []
    for i, conversation in enumerate(batch_data):
        task = predict_single_conversation(conversation)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    batch_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            batch_results.append({"error": f"异常: {str(result)}"})
        else:
            batch_results.append(result)

    return batch_results


async def main():
    output_file = "predicted_labels.jsonl"
    batch_size = 20  # 每批处理的数据量

    # 读取测试数据
    url = 'https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250616/ssrgii/test.jsonl'
    conversations = []

    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        for line_num, line in enumerate(response.text.splitlines(), 1):
            try:
                data = json.loads(line.strip())
                conversations.append(data)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                continue
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return

    print(f"成功读取 {len(conversations)} 条对话数据")

    # 分批处理
    all_results = []
    total_batches = (len(conversations) + batch_size - 1) // batch_size

    for batch_id in range(total_batches):
        start_idx = batch_id * batch_size
        end_idx = min((batch_id + 1) * batch_size, len(conversations))
        batch_data = conversations[start_idx:end_idx]

        batch_results = await process_batch(batch_data, batch_id + 1)
        all_results.extend(batch_results)

        print(f"批次 {batch_id + 1}/{total_batches} 完成")

        # 添加小延迟避免请求过快
        if batch_id < total_batches - 1:
            await asyncio.sleep(1)

    # 保存结果
    same_count = 0
    different_count = 0
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (original_data, prediction_result) in enumerate(zip(conversations, all_results)):
            result_entry = {
                "index": i,
                "original_user_content": None,
                "actual_address": None,
                "predicted_address": None,
                "prediction_error": None,
                "address_comparison": None
            }

            # 提取原始用户内容
            messages = original_data.get("messages", [])
            for message in messages:
                if message.get("role") == "user":
                    result_entry["original_user_content"] = message.get("content", "")
                    break

            # 提取实际地址信息（如果存在assistant message）
            for message in messages:
                if message.get("role") == "assistant":
                    result_entry["actual_address"] = message.get("content", "")
                    break

            # 保存预测结果
            if "error" in prediction_result:
                result_entry["prediction_error"] = prediction_result["error"]
                error_count += 1
            else:
                result_entry["predicted_address"] = prediction_result.get("prediction", "")

                # 比较地址信息
                comparison_result = compare_address_info(
                    result_entry["actual_address"],
                    result_entry["predicted_address"]
                )
                result_entry["address_comparison"] = comparison_result

                # 统计比较结果
                if comparison_result["comparison_error"]:
                    error_count += 1
                elif comparison_result["is_same"]:
                    same_count += 1
                else:
                    different_count += 1

            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

    print(f"所有预测完成! 结果已保存到 {output_file}")

    # 统计结果
    success_count = sum(1 for result in all_results if "error" not in result)
    prediction_error_count = len(all_results) - success_count
    print(f"样本数: {success_count} 条")
    print(f"响应正确: {same_count} 条")
    print(f"响应错误: {different_count} 条")
    print(f"准确率: {same_count * 100 / success_count} %")


if __name__ == "__main__":
    asyncio.run(main())