from pathlib import Path
import json
import pandas as pd

# 需要修改的两个地方
# 1. 换成你自己的 JSON 文件路径
# 2. 如果处理的不是 constantPositionOffset，就改成对应的攻击子集名称
file_path = Path(r"把这里换成你的 JSON 文件路径")
source_attack_subset = "constantPositionOffset"


def to_float(value):
    """把字符串或数字转换成浮点数；失败时返回 None。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def split_xyz(value):
    """把 'x,y,z' 字符串拆成三个浮点数。"""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = []

    if len(parts) < 3:
        return None, None, None

    return to_float(parts[0]), to_float(parts[1]), to_float(parts[2])


def flatten_message(message, source_attack_subset):
    """把一条嵌套 JSON 消息展开成一行。"""
    receiver = message.get("receiver", {})
    sender = message.get("sender", {})

    receiver_pos_x, receiver_pos_y, receiver_pos_z = split_xyz(receiver.get("pos"))
    receiver_pos_noise_x, receiver_pos_noise_y, receiver_pos_noise_z = split_xyz(receiver.get("pos_noise"))
    sender_pos_x, sender_pos_y, sender_pos_z = split_xyz(sender.get("pos"))
    sender_pos_noise_x, sender_pos_noise_y, sender_pos_noise_z = split_xyz(sender.get("pos_noise"))

    rcv_time = message.get("rcvTime")
    send_time = message.get("sendTime")
    delay_ns = None
    if rcv_time is not None and send_time is not None:
        delay_ns = int(rcv_time) - int(send_time)

    if all(value is not None for value in [receiver_pos_x, receiver_pos_y, receiver_pos_z,
                                           sender_pos_x, sender_pos_y, sender_pos_z]):
        sender_receiver_distance_m = (
            (sender_pos_x - receiver_pos_x) ** 2
            + (sender_pos_y - receiver_pos_y) ** 2
            + (sender_pos_z - receiver_pos_z) ** 2
        ) ** 0.5
    else:
        sender_receiver_distance_m = None

    attacker_raw = message.get("attacker")
    if attacker_raw is None and source_attack_subset == "baseline":
        is_attack = 0
        attacker_source = "baseline_assigned_zero"
    elif attacker_raw is None:
        is_attack = None
        attacker_source = "missing_in_message"
    else:
        is_attack = int(attacker_raw)
        attacker_source = "message_field"

    if is_attack == 1:
        attack_type = source_attack_subset
    elif is_attack == 0:
        attack_type = "benign"
    else:
        attack_type = None

    return {
        "source_attack_subset": source_attack_subset,
        "rcvTime": rcv_time,
        "sendTime": send_time,
        "delay_ns": delay_ns,
        "sender_id": message.get("sender_id"),
        "sender_alias": message.get("sender_alias"),
        "messageID": message.get("messageID"),
        "attacker_raw": attacker_raw,
        "attacker_source": attacker_source,
        "is_attack": is_attack,
        "attack_type": attack_type,

        "receiver_pos": receiver.get("pos"),
        "receiver_pos_x": receiver_pos_x,
        "receiver_pos_y": receiver_pos_y,
        "receiver_pos_z": receiver_pos_z,
        "receiver_pos_noise": receiver.get("pos_noise"),
        "receiver_pos_noise_x": receiver_pos_noise_x,
        "receiver_pos_noise_y": receiver_pos_noise_y,
        "receiver_pos_noise_z": receiver_pos_noise_z,
        "receiver_spd": to_float(receiver.get("spd")),
        "receiver_spd_noise": to_float(receiver.get("spd_noise")),
        "receiver_acl": to_float(receiver.get("acl")),
        "receiver_acl_noise": to_float(receiver.get("acl_noise")),
        "receiver_hed": to_float(receiver.get("hed")),
        "receiver_hed_noise": to_float(receiver.get("hed_noise")),
        "receiver_driversProfile": receiver.get("driversProfile"),

        "sender_pos": sender.get("pos"),
        "sender_pos_x": sender_pos_x,
        "sender_pos_y": sender_pos_y,
        "sender_pos_z": sender_pos_z,
        "sender_pos_noise": sender.get("pos_noise"),
        "sender_pos_noise_x": sender_pos_noise_x,
        "sender_pos_noise_y": sender_pos_noise_y,
        "sender_pos_noise_z": sender_pos_noise_z,
        "sender_spd": to_float(sender.get("spd")),
        "sender_spd_noise": to_float(sender.get("spd_noise")),
        "sender_acl": to_float(sender.get("acl")),
        "sender_acl_noise": to_float(sender.get("acl_noise")),
        "sender_hed": to_float(sender.get("hed")),
        "sender_hed_noise": to_float(sender.get("hed_noise")),
        "sender_driversProfile": sender.get("driversProfile"),
        "sender_distance_to_road_edge": to_float(sender.get("distance_to_road_edge")),
        "sender_receiver_distance_m": sender_receiver_distance_m,
    }


# 1. 读取 JSON 文件
with file_path.open("r", encoding="utf-8") as f:
    messages = json.load(f)

# 2. 循环处理所有消息
rows = []
for message in messages:
    row = flatten_message(message, source_attack_subset)
    rows.append(row)

# 3. 转换成 DataFrame
df_all = pd.DataFrame(rows)

# 4. 保存结果
output_path = file_path.with_name(file_path.stem + "_整理后.csv")
df_all.to_csv(output_path, index=False, encoding="utf-8-sig")

# 5. 输出检查结果
print("输入文件：", file_path)
print("输出文件：", output_path)
print("表格形状：", df_all.shape)
print("字段数量：", len(df_all.columns))
print("\nattacker 分布：")
print(df_all["is_attack"].value_counts(dropna=False))
print("\nattack_type 分布：")
print(df_all["attack_type"].value_counts(dropna=False))
print("\n每列缺失值数量：")
print(df_all.isna().sum())
print("\n前 3 行：")
print(df_all.head(3))
