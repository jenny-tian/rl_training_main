"""Fill the SJTU patent disclosure template with the current Lite3 project."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
TEMPLATE_DOC = Path("/home/tjx/Documents/组会/（交大）专利技术交底书模板-20240401.doc")
TEMPLATE_DOCX = ROOT / "reference_template" / "（交大）专利技术交底书模板-20240401.docx"
OUTPUT = ROOT / "Lite3_专利技术交底书_交大模板.docx"
FIG_DIR = ROOT / "disclosure_figures"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W}

TITLE = "一种基于潜变量残差控制的四足机器人在线地形适应方法"

BACKGROUND = [
    "四足机器人通常先在平地或仿真训练地形上学习站立、步态转换和速度跟踪策略，再将策略部署到崎岖、松软或具有高度起伏的地面。由于部署地形与训练分布存在差异，固定策略可能出现机身横向漂移、速度跟踪误差增加、足端接触异常以及跌倒等问题。",
    "现有强化学习方案可以通过目标地形继续训练策略，但若直接更新完整策略网络，有限的目标地形数据可能改变已学得的平地运动能力；若直接学习全关节动作残差，残差维数高，且动作容易在相邻控制周期突变。低秩参数适配可以减少可训练参数，但仍需解决如何利用历史本体观测形成适应信息、如何限制更新对基础策略的影响等问题。",
    "机器人控制链路还包括动作限幅、变化率限制、关节范围裁剪和接触保护。策略产生的请求动作经过这些模块后才成为实际执行动作。如果训练样本只保存其中一种动作而未记录二者差异，策略优化所依据的动作语义与环境实际执行不一致，可能影响更新和安全诊断。",
    "此外，候选策略需要同时检查目标地形适应效果与源平地保持能力。若缺少统一的候选验收、违规拒绝和检查点恢复流程，训练期间发生过安全违规的候选仍可能被后续探针错误晋升；开发选择与封存测试混用，也会使测试结果参与模型选择。",
    "因此，需要一种以已有平地策略为基础、仅在受限参数或低维残差空间中进行地形适应，并对请求动作、执行动作及候选策略状态进行一致记录和验证的方法。",
]

INNOVATIONS = [
    (
        "主要问题：直接微调完整策略容易改变原有平地技能；全维残差控制维数高、动作变化难约束，有限在线数据下适应方向不稳定。",
        "主要创新点：冻结源平地策略主体，在受限低秩参数空间中更新策略；并提供由本体历史和积分上下文产生低维 steering code、再通过固定低秩解码生成关节残差的实施方式。",
        "基本策略权重保持不变，在线优化只作用于指定低秩适配参数及其独立价值模块。潜变量路径将时序观测压缩为低维控制码，通过固定解码矩阵映射到多关节残差；残差幅值及相邻周期变化率受限。此结构减少在线更新自由度，并保留基础策略动作作为控制基准。",
        "降低需要在线更新的参数规模和动作自由度，限制策略漂移范围；基础策略动作可持续提供运动基准，适配部分可按配置启用或关闭。具体性能改善需通过源地与目标地形对照实验验证。",
    ),
    (
        "次要问题：请求动作经安全保护后可能改变为另一执行动作；若样本、策略概率和环境转移使用的动作语义混淆，会妨碍更新和违规原因追溯。",
        "次要创新点：显式区分请求动作与执行动作，记录投影差异及安全状态，并将该差异用于安全惩罚、样本筛选或候选拒绝。",
        "将策略提出的请求动作 a_req 输入安全投影器，得到实际发送给执行器的 a_exec；同一交互记录保存 a_req、a_exec、a_exec-a_req、投影原因和策略版本。环境转移按执行动作发生，策略分布仍按请求动作定义，安全诊断同时使用投影差异。",
        "使策略更新、环境实际动作和安全诊断之间的关系可审计；当投影差异超过设定范围时，可拒绝相应候选或过滤相关样本，便于定位动作限制触发原因。",
    ),
    (
        "次要问题：只根据训练回报选择策略不能确保平地能力保持或候选安全；训练、开发与封存测试数据混用会造成选择偏差。",
        "次要创新点：对候选策略执行源平地保持、目标地形冻结探针和安全违规检查；训练期违规候选禁止晋升恢复点，并隔离封存测试入口。",
        "候选发布前依次进行参数有限值与范围检查、策略漂移或 KL 检查、源平地行为锚定、目标地形冻结评估和安全违规审计。只有满足预设条件且无训练期安全违规的候选可更新恢复检查点。恢复时同步 Actor、Critic、优化器状态及校验哈希。封存测试只读取预先选定检查点并关闭更新。",
        "避免单次训练回报作为唯一晋升依据；使候选拒绝和恢复依据可追溯，并降低封存测试信息回流训练选择过程的风险。",
    ),
]

PRACTICALITY = [
    "本方法面向四足机器人仿真训练、实验室地面测试和机器人控制器部署。基础策略、低秩适配器、本体观测缓存、安全投影器及候选检查点管理均可由现有机器人计算平台上的软件模块实现，不要求改变机器人机械结构。",
    "在线控制周期中，观测编码、低秩矩阵运算、低维残差解码和动作投影均为常规张量及向量运算；策略更新可按 rollout 批次进行，与高频执行控制分离。安全投影器在每一控制周期运行，更新过程可暂停，执行侧仍使用已验证策略。",
    "项目代码已实现 Direct LoRA-PPO、源地与目标地形行为锚定、请求/执行动作记录、候选拒绝与检查点回滚、本体积分上下文以及离线教师残差的固定低秩解码器和编码器监督接口。低维 latent-SAC 为初版接口；特权教师轨迹采集、完整动作反演数据集、高度扫描编码器训练和潜变量残差方案端到端效果尚未完成验证，本文将其作为可选实施方式描述。",
    "应用验证可在具有相同机器人模型和控制频率的仿真环境开展，记录成功率、速度跟踪误差、机身姿态、接触违规、动作投影差异、源平地保持误差和候选回滚次数；之后再进行实体机器人测试。当前交底不主张未经实验确认的数值性能或量产成本结论。",
]

EXAMPLES = [
    "实施例一：Direct LoRA-PPO 在线适配。加载在平地训练的 PPO Actor，将指定线性层替换为包含冻结基础权重和低秩增量的适配层。对每层可表示为 W_i=W_{i,0}+(α_i/r)B_iA_i，其中 W_{i,0} 冻结，A_i、B_i 为适配参数，r 为低秩维数。适配初始化时令 B_i 为零，使初始确定性动作与基础策略一致。项目当前配置在四个线性层注入 rank=2 的 LoRA；该数值是实施配置示例，不构成保护范围限制。",
    "控制时，机器人读取本体观测和速度指令，适配 Actor 产生请求动作。执行器之前的安全层依次进行动作幅值、变化率、关节目标范围和接触保护处理，输出执行动作。每条样本保存策略版本、请求动作、执行动作、二者差值、奖励、终止/截断标记和必要的安全诊断字段。",
    "每个 rollout 使用同一策略版本的数据计算优势，区分真实终止与时间截断，并使用终止前状态作价值引导。更新对象包括低秩适配参数和独立 Critic。更新过程中可加入冻结基础策略行为锚定、源平地保持约束及请求/执行差异惩罚。更新结束后生成候选检查点，而非直接覆盖最后一个已验证检查点。",
    "实施例二：本体积分上下文适配。基础 Actor 接收 45 维基础观测，适配 Actor 额外接收 3 维本体积分上下文。上下文由机身坐标系平面速度和偏航角速度进行中点积分，形成估计航向正弦量、航向余弦偏差及归一化累计横移量。积分过程只使用本体感知，不读取仿真世界位置、世界偏航角或预先提供的地形类别标签。",
    "当环境终止时，系统先保存终止前上下文以供末帧价值计算，之后清零积分器，避免复位后新回合状态污染上一回合样本。适配策略分别在源平地和目标地形的冻结探针上评估，开发地形用于候选选择，封存测试地形不参与适配和选择。",
    "实施例三：请求动作和执行动作一致性处理。令 a_req 为策略输出的请求动作，P_safe 为安全投影算子，则 a_exec=P_safe(a_req)。投影算子可包括动作范围裁剪、相邻周期变化率限制、关节目标范围限制及异常接触保护。交互缓冲区保存 (o_t,a_req,a_exec,a_exec-a_req,r_t,o_{t+1}) 及策略版本和终止标记。环境实际状态转移由 a_exec 产生；策略概率比和请求动作价值仍使用 a_req 的动作语义，避免将 a_exec 误当作策略采样动作。",
    "候选策略验收时统计投影差异及其原因。若安全投影持续改变请求动作或触发配置的拒绝条件，则候选不发布，并恢复最近一个冻结验证通过且无安全违规的检查点。恢复操作同步 Actor、Critic、Actor 优化器和 Critic 优化器状态，并更新检查点校验信息。",
    "实施例四：候选验证、恢复与测试隔离。候选策略依次通过模型参数有限值检查、参数幅值或近似 KL 检查、源平地行为锚定、目标地形冻结探针及安全违规审计。训练期一旦发生规定的安全违规，该候选即被标记为不可晋升恢复检查点，即使后续冻结探针暂时通过也不改变该标记。系统回到最近的无违规冻结验证检查点继续运行。",
    "地形种子及初始条件按训练集、开发验证集和封存测试集划分。训练集允许参数更新；开发集用于预先规定的候选选择；封存测试入口仅加载预先选定的 checkpoint，关闭梯度与参数写入，不修改训练状态，并生成独立的测试清单、协议哈希、检查点哈希和逐场景结果。",
    "实施例五：可选的潜变量残差路径。该路径将本体历史和可用地形摘要编码为潜变量 z_t，再生成低维 tanh 控制码 u_t，经固定解码矩阵映射为多关节残差，并逐关节限制残差幅值及控制周期变化率。残差叠加到冻结基础策略动作后进入同一安全投影器。可选地启用状态风险缩放：将姿态偏差、机身高度、关节速度和非足接触状态分别归一化并取最大值，按有界线性函数缩小残差包络；硬安全暂停时残差置零。该安全缩放为可选安全特征，不作为单独的核心创新主张。",
    "固定低秩解码矩阵可由按关节动作限幅归一化的教师残差样本通过截断奇异值分解拟合，并绑定动作限幅、来源协议、地形划分、重构误差和解释方差等元数据。若使用特权教师动作求解潜变量监督目标，应单独记录数据来源及生成过程；相关教师采集和端到端训练属于可选实施方式，项目尚无完整验证结果。",
]


def q(name: str) -> str:
    return f"{{{W}}}{name}"


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_figure(path: Path, title: str, labels: list[str]) -> None:
    image = Image.new("RGB", (1600, 950), "white")
    d = ImageDraw.Draw(image)
    ink = "black"
    d.text((800, 55), title, fill=ink, font=font(38), anchor="mm")
    d.line((70, 100, 1530, 100), fill=ink, width=2)
    xs = [90, 465, 840, 1215]
    y_top, y_bottom, bw, bh = 200, 570, 290, 145
    boxes = []
    for i, label in enumerate(labels):
        row, col = divmod(i, 4)
        x = xs[col]
        y = y_top if row == 0 else y_bottom
        d.rectangle((x, y, x + bw, y + bh), outline=ink, width=3)
        d.ellipse((x + 14, y + 14, x + 52, y + 52), outline=ink, width=2)
        d.text((x + 33, y + 33), str(i + 1), fill=ink, font=font(22), anchor="mm")
        lines = label.split("\n")
        for j, line in enumerate(lines):
            d.text((x + bw // 2, y + 70 + j * 34), line, fill=ink, font=font(25), anchor="mm")
        boxes.append((x, y, x + bw, y + bh))
    # The process runs across the first row, drops at the right, then returns.
    for left, right in zip(boxes[:4], boxes[1:4]):
        cy = y_top + bh // 2
        d.line((left[2], cy, right[0] - 12, cy), fill=ink, width=3)
        d.polygon(((right[0] - 15, cy - 8), (right[0] - 2, cy), (right[0] - 15, cy + 8)), fill=ink)
    x = boxes[3][0] + bw // 2
    d.line((x, boxes[3][3], x, y_bottom - 18), fill=ink, width=3)
    d.polygon(((x - 8, y_bottom - 22), (x, y_bottom - 6), (x + 8, y_bottom - 22)), fill=ink)
    for left, right in zip(boxes[4:], boxes[5:]):
        cy = y_bottom + bh // 2
        d.line((right[0], cy, left[2] + 12, cy), fill=ink, width=3)
        d.polygon(((left[2] + 15, cy - 8), (left[2] + 2, cy), (left[2] + 15, cy + 8)), fill=ink)
    image.save(path)


def generate_figures() -> list[Path]:
    FIG_DIR.mkdir(exist_ok=True)
    specs = [
        ("图1 四足机器人在线地形适应方法总体流程", ["加载冻结平地策略", "采集本体观测\n构造时序上下文", "低秩适配或\n潜变量残差生成", "安全投影\n发送执行动作", "记录请求/执行\n动作及反馈", "更新适配参数\n与价值模块", "源地保持与\n目标地形探针", "通过则发布\n违规则回滚"]),
        ("图2 请求动作与执行动作一致性处理流程", ["时序观测与\n速度指令", "基础策略\n生成基准动作", "适配模块\n生成请求动作", "安全投影器\n幅值/速率/关节", "执行器\n接收执行动作", "环境状态\n产生转移", "记录动作差值\n与安全原因", "更新诊断\n或拒绝候选"]),
        ("图3 候选策略验证、回滚与数据隔离流程", ["训练数据\n允许参数更新", "生成候选\n保存完整状态", "参数与漂移\n约束检查", "源地保持与\n目标地冻结探针", "安全审计\n检查训练违规", "通过候选\n登记校验哈希", "违规候选\n禁止晋升", "恢复最近有效\n检查点/封存测试"]),
    ]
    paths = []
    for i, (title, labels) in enumerate(specs, 1):
        p = FIG_DIR / f"figure{i}.png"
        make_figure(p, title, labels)
        paths.append(p)
    return paths


def set_cell_text(cell, paragraphs: list[str]) -> None:
    old = cell.findall("w:p", NS)
    template = old[0] if old else E.Element(q("p"))
    ppr = template.find("w:pPr", NS)
    ppr_copy = E.fromstring(E.tostring(ppr)) if ppr is not None else None
    for child in list(cell):
        if child.tag == q("p"):
            cell.remove(child)
    for text in paragraphs:
        p = E.Element(q("p"))
        if ppr_copy is not None:
            p.append(E.fromstring(E.tostring(ppr_copy)))
        r = E.SubElement(p, q("r"))
        rpr = E.SubElement(r, q("rPr"))
        E.SubElement(rpr, q("rFonts"), {q("ascii"): "Times New Roman", q("hAnsi"): "Times New Roman", q("eastAsia"): "宋体"})
        E.SubElement(rpr, q("sz"), {q("val"): "21"})
        E.SubElement(rpr, q("szCs"), {q("val"): "21"})
        t = E.SubElement(r, q("t"))
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
        cell.append(p)
    if not paragraphs:
        cell.append(E.Element(q("p")))


def add_image_paragraph(cell, rid: str, docpr_id: int, cx: int, cy: int) -> None:
    p = E.SubElement(cell, q("p"))
    ppr = E.SubElement(p, q("pPr"))
    E.SubElement(ppr, q("jc"), {q("val"): "center"})
    run = E.SubElement(p, q("r"))
    drawing = E.SubElement(run, q("drawing"))
    inline = E.SubElement(drawing, f"{{{WP}}}inline", {"distT": "0", "distB": "0", "distL": "0", "distR": "0"})
    E.SubElement(inline, f"{{{WP}}}extent", {"cx": str(cx), "cy": str(cy)})
    E.SubElement(inline, f"{{{WP}}}docPr", {"id": str(docpr_id), "name": f"Disclosure Figure {docpr_id}"})
    E.SubElement(inline, f"{{{WP}}}cNvGraphicFramePr")
    graphic = E.SubElement(inline, f"{{{A}}}graphic")
    gd = E.SubElement(graphic, f"{{{A}}}graphicData", {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"})
    pic = E.SubElement(gd, f"{{{PIC}}}pic")
    nv = E.SubElement(pic, f"{{{PIC}}}nvPicPr")
    E.SubElement(nv, f"{{{PIC}}}cNvPr", {"id": str(docpr_id), "name": f"Figure{docpr_id}.png"})
    E.SubElement(nv, f"{{{PIC}}}cNvPicPr")
    fill = E.SubElement(pic, f"{{{PIC}}}blipFill")
    E.SubElement(fill, f"{{{A}}}blip", {f"{{{R}}}embed": rid})
    stretch = E.SubElement(fill, f"{{{A}}}stretch")
    E.SubElement(stretch, f"{{{A}}}fillRect")
    sp = E.SubElement(pic, f"{{{PIC}}}spPr")
    xfrm = E.SubElement(sp, f"{{{A}}}xfrm")
    E.SubElement(xfrm, f"{{{A}}}off", {"x": "0", "y": "0"})
    E.SubElement(xfrm, f"{{{A}}}ext", {"cx": str(cx), "cy": str(cy)})
    geom = E.SubElement(sp, f"{{{A}}}prstGeom", {"prst": "rect"})
    E.SubElement(geom, f"{{{A}}}avLst")


def main() -> None:
    if not TEMPLATE_DOCX.exists():
        TEMPLATE_DOCX.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sjtu_disclosure_template_") as profile:
            subprocess.run(["libreoffice", "-env:UserInstallation=file://" + profile, "--headless", "--convert-to", "docx", "--outdir", str(TEMPLATE_DOCX.parent), str(TEMPLATE_DOC)], check=True, timeout=120)
    figures = generate_figures()
    with ZipFile(TEMPLATE_DOCX) as z:
        package = {item.filename: z.read(item.filename) for item in z.infolist()}
    doc = E.fromstring(package["word/document.xml"])
    body = doc.find(".//w:body", NS)
    tables = body.findall("w:tbl", NS)
    if len(tables) < 7:
        raise RuntimeError("The SJTU template table layout has changed")
    title_cell = tables[0].find(".//w:tc", NS)
    set_cell_text(title_cell, [TITLE])
    bg_cell = tables[1].find(".//w:tc", NS)
    set_cell_text(bg_cell, BACKGROUND)
    innovation_rows = tables[2].findall("w:tr", NS)
    # Keep each innovation as one readable unit when Word paginates the table.
    header_pr = innovation_rows[0].find("w:trPr", NS)
    if header_pr is None:
        header_pr = E.Element(q("trPr"))
        innovation_rows[0].insert(0, header_pr)
    E.SubElement(header_pr, q("tblHeader"))
    for row in innovation_rows:
        tr_pr = row.find("w:trPr", NS)
        if tr_pr is None:
            tr_pr = E.Element(q("trPr"))
            row.insert(0, tr_pr)
        if tr_pr.find("w:cantSplit", NS) is None:
            E.SubElement(tr_pr, q("cantSplit"))
    for row, data in zip(innovation_rows[1:4], INNOVATIONS):
        cells = row.findall("w:tc", NS)
        for cell, value in zip(cells, data):
            set_cell_text(cell, [value])
    # Remove the unused ellipsis row from the template.
    if len(innovation_rows) > 4:
        tables[2].remove(innovation_rows[4])
    practical_cell = tables[3].find(".//w:tc", NS)
    set_cell_text(practical_cell, PRACTICALITY)
    example_cell = tables[4].find(".//w:tc", NS)
    set_cell_text(example_cell, EXAMPLES)
    figure_cell = tables[5].find(".//w:tc", NS)
    set_cell_text(figure_cell, ["图1 四足机器人在线地形适应方法总体流程", "图2 请求动作与执行动作一致性处理流程", "图3 候选策略验证、回滚与数据隔离流程", "附图中方框表示处理步骤，箭头表示数据或控制流程；图中缩写沿用说明书定义。"])
    relroot = E.fromstring(package["word/_rels/document.xml.rels"])
    for i, figure in enumerate(figures, 1):
        rid = f"rIdDisclosure{i}"
        relroot.append(E.Element(f"{{{PKG_REL}}}Relationship", {"Id": rid, "Type": f"{R}/image", "Target": f"media/disclosure_figure{i}.png"}))
        with Image.open(figure) as im:
            cx = 5_800_000
            cy = int(cx * im.height / im.width)
        add_image_paragraph(figure_cell, rid, 100 + i, cx, cy)
        package[f"word/media/disclosure_figure{i}.png"] = figure.read_bytes()
    sequence_cell = tables[6].find(".//w:tc", NS)
    set_cell_text(sequence_cell, ["本项目不涉及生物材料，也不涉及核酸或氨基酸序列。"])
    ct = E.fromstring(package["[Content_Types].xml"])
    if not any(node.get("Extension") == "png" for node in ct.findall(f"{{{CT}}}Default")):
        ct.append(E.Element(f"{{{CT}}}Default", {"Extension": "png", "ContentType": "image/png"}))
    package["[Content_Types].xml"] = E.tostring(ct, encoding="UTF-8", xml_declaration=True, standalone=True)
    # Add one dynamic, centered page number field.  PAGE is evaluated by Word
    # or LibreOffice, so it remains correct when the document repaginates.
    footer_rid = "rIdDisclosureFooter"
    relroot.append(E.Element(f"{{{PKG_REL}}}Relationship", {
        "Id": footer_rid,
        "Type": f"{R}/footer",
        "Target": "footerDisclosure.xml",
    }))
    footer = E.Element(f"{{{W}}}ftr", nsmap={"w": W})
    fp = E.SubElement(footer, q("p"))
    fppr = E.SubElement(fp, q("pPr"))
    E.SubElement(fppr, q("jc"), {q("val"): "center"})
    fr = E.SubElement(fp, q("r"))
    frpr = E.SubElement(fr, q("rPr"))
    E.SubElement(frpr, q("rFonts"), {q("ascii"): "Times New Roman", q("hAnsi"): "Times New Roman", q("eastAsia"): "宋体"})
    E.SubElement(frpr, q("sz"), {q("val"): "21"})
    E.SubElement(frpr, q("szCs"), {q("val"): "21"})
    E.SubElement(fr, q("fldChar"), {q("fldCharType"): "begin"})
    instr = E.SubElement(fr, q("instrText"))
    instr.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    instr.text = " PAGE "
    E.SubElement(fr, q("fldChar"), {q("fldCharType"): "separate"})
    result = E.SubElement(fr, q("t"))
    result.text = "1"
    E.SubElement(fr, q("fldChar"), {q("fldCharType"): "end"})
    package["word/footerDisclosure.xml"] = E.tostring(footer, encoding="UTF-8", xml_declaration=True, standalone=True)
    if not any(node.get("PartName") == "/word/footerDisclosure.xml" for node in ct.findall(f"{{{CT}}}Override")):
        ct.append(E.Element(f"{{{CT}}}Override", {"PartName": "/word/footerDisclosure.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"}))
    package["[Content_Types].xml"] = E.tostring(ct, encoding="UTF-8", xml_declaration=True, standalone=True)
    sect_pr = body.find("w:sectPr", NS)
    if sect_pr is not None and sect_pr.find(f"w:footerReference[@r:id='{footer_rid}']", {**NS, "r": R}) is None:
        sect_pr.insert(0, E.Element(q("footerReference"), {q("type"): "default", f"{{{R}}}id": footer_rid}))
    package["word/_rels/document.xml.rels"] = E.tostring(relroot, encoding="UTF-8", xml_declaration=True, standalone=True)
    # Delete only the instructional helper paragraphs below section three.
    for p in list(body.findall("w:p", NS)):
        text = "".join(p.xpath(".//w:t/text()", namespaces=NS)).strip()
        if text.startswith("【1.“问题-") or text.startswith("2. 一项发明可能有多个") or text.startswith("3.“技术效果”包括") or text.startswith("4.“古怪的”"):
            body.remove(p)
    package["word/document.xml"] = E.tostring(doc, encoding="UTF-8", xml_declaration=True, standalone=True)
    with ZipFile(OUTPUT, "w") as z:
        for name, data in package.items():
            z.writestr(name, data)
    print(OUTPUT)


if __name__ == "__main__":
    main()
