"""Sector-name normalization and grouping."""


GROUP_KEYWORDS = {
    "有色资源类": [
        "有色", "金属", "贵金属", "小金属", "能源金属", "稀土", "煤炭",
        "石油", "化工", "化肥", "钢铁", "黄金", "矿",
    ],
    "半导体": ["半导体", "芯片", "集成电路"],
    "元器件": ["元件", "电子元件", "消费电子", "光学光电子", "PCB", "印制电路"],
    "通信设备": ["通信设备", "通信服务", "光通信", "5G"],
    "电气设备": ["电池", "电源设备", "光伏", "风电", "电网", "电机", "电气", "电力设备"],
}


def classify_group(name):
    text = str(name)
    for group, keywords in GROUP_KEYWORDS.items():
        if any(key in text for key in keywords):
            return group
    return "其他板块"


def normalize_board_name(name):
    return str(name).replace("行业板块", "").strip()
