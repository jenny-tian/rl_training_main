"""Build the patent application using the supplied Chinese patent layout."""

from __future__ import annotations

import hashlib
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "patent.md"
REFERENCE = Path("/home/tjx/Documents/组会/RE2-J0592-24003PIX#J-一种基于模仿学习的巡检机器人避障算法.doc")
STEM = "Lite3_在线地形适应发明专利_参考格式_20260923"
FIGURES = ROOT / "figures"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": W}


def q(name: str) -> str:
    return "{" + W + "}" + name


def section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1 : end]).strip()


def patent_title(markdown: str) -> str:
    value = section(markdown, "## 一、发明名称")
    return value.strip().rstrip("。")


def claims(markdown: str) -> str:
    body = section(markdown, "## 九、权利要求书")
    pieces = re.split(r"(?m)^### 权利要求 (\d+)\s*$", body)
    rendered = []
    for index in range(1, len(pieces), 2):
        number = pieces[index]
        text = pieces[index + 1].strip()
        text = re.sub(r"\n{2,}", "\n\n", text)
        rendered.append(f"**{number}、**{text}")
    return "\n\n".join(rendered)


def build_application_source(markdown: str) -> str:
    title = patent_title(markdown)
    abstract = section(markdown, "## 十、摘要")
    abstract = re.sub(r"(?m)^关键词：.*$", "", abstract).strip()
    purpose = section(markdown, "## 四、发明目的")
    scheme = section(markdown, "## 五、技术方案")
    effects = section(markdown, "## 六、有益效果")
    figure_description = section(markdown, "## 七、附图说明")
    examples = section(markdown, "## 八、具体实施例")
    # Keep the application body focused on disclosure. This internal status note
    # is preserved as an implementation boundary, not as a claimed result.
    status = re.search(r"本项目现有代码已实现.*?既有实验结果。", effects, re.S)
    status_note = status.group(0) if status else ""
    if status:
        effects = effects[: status.start()].rstrip()

    application = f'''# 说 明 书 摘 要

{abstract}

# 摘 要 附 图

![图1 本发明在线地形适应方法总体流程图](figures/figure1_overview.png){{width=90%}}

# 权 利 要 求 书

{claims(markdown)}

# 说 明 书

## {title}

### 技术领域

{section(markdown, "## 二、技术领域")}

### 背景技术

{section(markdown, "## 三、背景技术")}

### 发明内容

#### 发明目的

{purpose}

#### 技术方案

{scheme}

#### 有益效果

{effects}

{status_note}

### 附图说明

{figure_description}

### 具体实施方式

以下结合附图和实施例对本发明作进一步说明。以下实施例用于解释本发明，不构成对保护范围的限定。

{examples}

### 实现状态及实施边界

本项目当前实现包括直接 LoRA-PPO、source-flat/terrain anchor、请求动作与执行动作记录、候选拒绝和策略回滚、终止前状态捕获、3 维本体积分上下文、第一版低维 latent-SAC 接口，以及离线教师残差固定低秩解码器和编码器监督接口。真实特权教师轨迹采集、完整动作反演数据集、以高度扫描作为策略 Actor 输入的编码器训练，以及 RGLS 端到端性能验证尚未完成。本说明书将这些内容作为可选实施方式描述，不将其表述为已经取得的项目实验结果。

FlowDAgger 的潜空间动作反演思想可用于设计教师动作到低维 steering code 的监督过程；本项目基础控制策略为 PPO，不是 flow 或 diffusion policy，因此本方案不称为 FlowDAgger，也不主张复现 FlowDAgger。

# 说 明 书 附 图

![图1 本发明在线地形适应方法总体流程图](figures/figure1_overview.png){{width=92%}}

# 说 明 书 附 图（续）

![图2 请求动作、安全投影及执行动作关系图](figures/figure2_action_safety.png){{width=92%}}

![图3 候选策略验证、回滚和数据隔离流程图](figures/figure3_validation_split.png){{width=92%}}
'''
    # Put each formal component on a new page, matching the supplied filing.
    return application


def figure_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_diagram(path: Path, title: str, rows: list[list[str]], notes: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1800, 1200), "white")
    draw = ImageDraw.Draw(image)
    title_font = figure_font(42)
    box_font = figure_font(29)
    note_font = figure_font(25)
    step_font = figure_font(22)
    ink = "#26363D"
    accent = "#176B67"
    box_fill = "#F4F7F7"
    soft_fill = "#E6F0EF"
    draw.text((900, 58), title, font=title_font, fill=ink, anchor="mm")
    draw.line((85, 105, 1715, 105), fill="#AEBDBC", width=2)
    x_gap = 34
    margin = 70
    y0 = 180
    row_gap = 300
    box_height = 152
    layout: list[list[tuple[int, int, int, int]]] = []
    for row_index, labels in enumerate(rows):
        count = len(labels)
        box_width = (1800 - 2 * margin - x_gap * (count - 1)) // count
        y = y0 + row_index * row_gap
        positions = []
        visual_indices = list(range(count)) if row_index == 0 else list(reversed(range(count)))
        for visual_index, index in enumerate(visual_indices):
            label = labels[index]
            x1 = margin + visual_index * (box_width + x_gap)
            x2 = x1 + box_width
            y2 = y + box_height
            draw.rounded_rectangle((x1, y, x2, y2), radius=12, outline=ink, width=3, fill=box_fill)
            draw.rounded_rectangle((x1, y, x2, y + 12), radius=6, outline=None, fill=accent)
            draw.ellipse((x1 + 14, y + 20, x1 + 52, y + 58), fill=soft_fill, outline=accent, width=2)
            step_number = index + (1 if row_index == 0 else 6)
            draw.text((x1 + 33, y + 39), f"{step_number:02d}", font=step_font, fill=accent, anchor="mm")
            lines = label.split("\n")
            line_height = 38
            start_y = y + 72 + (box_height - 78 - len(lines) * line_height) / 2
            for line_no, line in enumerate(lines):
                draw.text(((x1 + x2) / 2, start_y + line_no * line_height), line, font=box_font, fill=ink, anchor="mm")
            positions.append((x1, x2, y, y2))
        layout.append(positions)
        ordered = sorted(positions, key=lambda rect: rect[0])
        pairs = zip(ordered, ordered[1:]) if row_index == 0 else zip(reversed(ordered[1:]), reversed(ordered[:-1]))
        for left, right in pairs:
            if row_index == 0:
                sx, ex = left[1], right[0]
                cy = y + box_height // 2
                draw.line((sx + 3, cy, ex - 13, cy), fill=accent, width=4)
                draw.polygon(((ex - 14, cy - 10), (ex, cy), (ex - 14, cy + 10)), fill=accent)
            else:
                sx, ex = left[0], right[1]
                cy = y + box_height // 2
                draw.line((sx - 3, cy, ex + 13, cy), fill=accent, width=4)
                draw.polygon(((ex + 14, cy - 10), (ex, cy), (ex + 14, cy + 10)), fill=accent)
    if len(layout) > 1 and layout[0] and layout[1]:
        top_right = max(layout[0], key=lambda rect: rect[0])
        bottom_right = max(layout[1], key=lambda rect: rect[0])
        x = (top_right[0] + top_right[1]) // 2
        y1 = top_right[3]
        y2 = bottom_right[2]
        draw.line((x, y1 + 2, x, y2 - 18), fill=accent, width=4)
        draw.polygon(((x - 10, y2 - 20), (x, y2 - 4), (x + 10, y2 - 20)), fill=accent)
    if notes:
        y = 930
        draw.line((85, y - 35, 1715, y - 35), fill="#D5DEDD", width=2)
        for note in notes:
            draw.text((900, y), note, font=note_font, fill="#52615F", anchor="mm")
            y += 46
    image.save(path, dpi=(220, 220))


def generate_figures() -> None:
    draw_diagram(
        FIGURES / "figure1_overview.png",
        "基于潜变量残差控制的四足机器人在线地形适应方法",
        [
            ["冻结平地 PPO\n基础策略", "本体历史与\n可选地形摘要", "低维 steering code\n固定低秩解码", "有界残差\n组合基础动作", "安全投影\n动作执行"],
            ["在线交互样本\n请求/执行记录", "LoRA-PPO 或\n低维 SAC 更新", "候选策略检查\n源地与目标地探针", "通过：发布\n违规：恢复"],
        ],
        ["基础策略参数保持冻结；仅允许受限适配参数更新。"],
    )
    draw_diagram(
        FIGURES / "figure2_action_safety.png",
        "请求动作、安全投影及执行动作关系",
        [
            ["冻结基础策略\na_base", "潜变量控制\n生成有界残差", "可选状态风险\n缩放残差", "请求动作\na_req", "安全投影器"],
            ["状态与接触\n风险信号", "动作幅值/变化率\n关节范围约束", "执行动作\na_exec", "投影差异\na_exec-a_req", "回放与更新\n安全诊断"],
        ],
        ["环境控制使用安全投影后的执行动作；请求动作语义单独保留用于策略更新。"],
    )
    draw_diagram(
        FIGURES / "figure3_validation_split.png",
        "候选策略验证、回滚和数据隔离流程",
        [
            ["训练种子\n参数更新", "候选 checkpoint\n完整状态与哈希", "有限值/KL/安全\n违规检查", "平地保持与\n目标地形冻结探针", "候选发布"],
            ["开发集\n仅用于选择", "失败候选\n拒绝并记录", "最近无违规\n验证 checkpoint", "封存测试入口\n冻结参数", "独立报告\n不得回流调参"],
        ],
        ["训练、开发和封存地形种子相互隔离；封存测试不参与策略选择。"],
    )


def convert_reference_doc(output_dir: Path) -> Path:
    if not REFERENCE.is_file():
        raise FileNotFoundError(f"reference patent document not found: {REFERENCE}")
    with tempfile.TemporaryDirectory(prefix="lite3_patent_template_") as profile:
        target_dir = output_dir / "reference_template"
        target_dir.mkdir(exist_ok=True)
        subprocess.run(
            [
                "libreoffice",
                "-env:UserInstallation=file://" + profile,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(target_dir),
                str(REFERENCE),
            ],
            check=True,
            timeout=120,
            stdout=subprocess.DEVNULL,
        )
    converted = target_dir / (REFERENCE.stem + ".docx")
    if not converted.is_file():
        raise RuntimeError("LibreOffice did not create the reference DOCX")
    return converted


def style_docx(path: Path) -> None:
    with ZipFile(path) as z:
        package = {item.filename: z.read(item.filename) for item in z.infolist()}
    document = E.fromstring(package["word/document.xml"])
    # The reference filing formats section labels and the invention title as
    # Normal paragraphs; Pandoc's default heading styles are substantially larger.
    normal_labels = {
        "技术领域", "背景技术", "发明内容", "发明目的", "技术方案", "有益效果",
        "附图说明", "具体实施方式", "实现状态及实施边界",
    }
    drawing_paragraphs = []
    page_titles = {
        "摘 要 附 图",
        "说 明 书 附 图",
        "说 明 书 附 图（续）",
    }
    active_figure_page = False
    figure_slot = 0
    for paragraph in document.findall(".//w:body/w:p", NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()
        if text in page_titles:
            active_figure_page = True
            figure_slot = 0
        elif text in {"权 利 要 求 书", "说 明 书", "说 明 书 附 图（续）"} and text not in page_titles:
            active_figure_page = False
        ppr = paragraph.find("w:pPr", NS)
        if ppr is None:
            ppr = E.Element(q("pPr"))
            paragraph.insert(0, ppr)
        pstyle = ppr.find("w:pStyle", NS)
        if text in normal_labels or (text.startswith("一种") and "四足机器人在线地形适应方法" in text):
            if pstyle is None:
                pstyle = E.SubElement(ppr, q("pStyle"))
            pstyle.set(q("val"), "Normal")
            jc = ppr.find("w:jc", NS)
            if jc is None:
                jc = E.SubElement(ppr, q("jc"))
            jc.set(q("val"), "center" if text.startswith("一种") else "both")
            for name in ("ind", "spacing", "keepNext", "keepLines", "outlineLvl"):
                node = ppr.find(f"w:{name}", NS)
                if node is not None:
                    ppr.remove(node)
            spacing = E.SubElement(ppr, q("spacing"))
            spacing.set(q("lineRule"), "exact")
            spacing.set(q("line"), "400")
            for run in paragraph.findall("w:r", NS):
                rpr = run.find("w:rPr", NS)
                if rpr is not None:
                    for name in ("sz", "szCs", "b", "bCs", "i", "iCs"):
                        node = rpr.find(f"w:{name}", NS)
                        if node is not None:
                            rpr.remove(node)
            if text in normal_labels:
                for run in paragraph.findall("w:r", NS):
                    rpr = run.find("w:rPr", NS)
                    if rpr is None:
                        rpr = E.Element(q("rPr"))
                        run.insert(0, rpr)
                    if rpr.find("w:u", NS) is None:
                        E.SubElement(rpr, q("u"), {q("val"): "single"})
        if paragraph.find(".//w:drawing", NS) is not None:
            drawing_paragraphs.append(paragraph)
            pstyle = ppr.find("w:pStyle", NS)
            if pstyle is None:
                pstyle = E.SubElement(ppr, q("pStyle"))
            pstyle.set(q("val"), "Normal")
            for name in ("ind", "keepNext", "keepLines", "outlineLvl"):
                node = ppr.find(f"w:{name}", NS)
                if node is not None:
                    ppr.remove(node)
            spacing = ppr.find("w:spacing", NS)
            if spacing is None:
                spacing = E.SubElement(ppr, q("spacing"))
            spacing.set(q("before"), "0")
            spacing.set(q("after"), "0")
            spacing.set(q("lineRule"), "exact")
            spacing.set(q("line"), "400")
            jc = ppr.find("w:jc", NS)
            if jc is None:
                jc = E.SubElement(ppr, q("jc"))
            jc.set(q("val"), "center")
            if active_figure_page:
                drawing = paragraph.find(".//w:drawing", NS)
                inline = drawing.find(f"{{{WP}}}inline")
                if inline is not None:
                    extent = inline.find(f"{{{WP}}}extent")
                    effect = inline.find(f"{{{WP}}}effectExtent")
                    docpr = inline.find(f"{{{WP}}}docPr")
                    frame_locks = inline.find(f"{{{WP}}}cNvGraphicFramePr")
                    graphic = inline.find("{http://schemas.openxmlformats.org/drawingml/2006/main}graphic")
                    anchor = E.Element(
                        f"{{{WP}}}anchor",
                        {
                            "distT": "0", "distB": "0", "distL": "0", "distR": "0",
                            "simplePos": "0", "relativeHeight": "0", "behindDoc": "0",
                            "locked": "0", "layoutInCell": "1", "allowOverlap": "1",
                        },
                    )
                    E.SubElement(anchor, f"{{{WP}}}simplePos", {"x": "0", "y": "0"})
                    pos_h = E.SubElement(anchor, f"{{{WP}}}positionH", {"relativeFrom": "page"})
                    E.SubElement(pos_h, f"{{{WP}}}posOffset").text = "1030000"
                    pos_v = E.SubElement(anchor, f"{{{WP}}}positionV", {"relativeFrom": "page"})
                    y_offset = 1450000 if figure_slot == 0 else 4750000
                    E.SubElement(pos_v, f"{{{WP}}}posOffset").text = str(y_offset)
                    if extent is not None:
                        anchor.append(extent)
                    if effect is not None:
                        anchor.append(effect)
                    E.SubElement(anchor, f"{{{WP}}}wrapNone")
                    if docpr is not None:
                        anchor.append(docpr)
                    if frame_locks is not None:
                        anchor.append(frame_locks)
                    if graphic is not None:
                        anchor.append(graphic)
                    drawing.remove(inline)
                    drawing.append(anchor)
                figure_slot += 1
    # Pandoc carries the template headers, footers, margins, numbering, and
    # typography. Add explicit page starts for the five filing components.
    page_break_titles = {
        "摘 要 附 图",
        "权 利 要 求 书",
        "说 明 书",
        "说 明 书 附 图",
        "说 明 书 附 图（续）",
    }
    for paragraph in document.findall(".//w:body/w:p", NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()
        if text not in page_break_titles:
            continue
        ppr = paragraph.find("w:pPr", NS)
        if ppr is None:
            ppr = E.Element(q("pPr"))
            paragraph.insert(0, ppr)
        if ppr.find("w:pageBreakBefore", NS) is None:
            E.SubElement(ppr, q("pageBreakBefore"))
        jc = ppr.find("w:jc", NS)
        if jc is None:
            jc = E.SubElement(ppr, q("jc"))
        jc.set(q("val"), "center")
    body = document.find(".//w:body", NS)
    for paragraph in list(body.findall("w:p", NS)):
        ppr = paragraph.find("w:pPr", NS)
        pstyle = ppr.find("w:pStyle", NS) if ppr is not None else None
        if pstyle is not None and pstyle.get(q("val")) == "ImageCaption":
            body.remove(paragraph)
    package["word/document.xml"] = E.tostring(document, encoding="UTF-8", xml_declaration=True, standalone=True)
    for name, data in list(package.items()):
        if not re.fullmatch(r"word/footer\d+\.xml", name):
            continue
        footer = E.fromstring(data)
        # The supplied template uses a narrow floating text box for the page
        # number. Two digit numbers then wrap vertically ("1" over "2").
        # Replace it with a normal centered PAGE field so 10, 11, 12, 13 stay
        # on one line and continue across the document.
        for child in list(footer):
            footer.remove(child)
        paragraph = E.SubElement(footer, q("p"))
        ppr = E.SubElement(paragraph, q("pPr"))
        E.SubElement(ppr, q("pStyle"), {q("val"): "Footer"})
        E.SubElement(ppr, q("jc"), {q("val"): "center"})
        field = E.SubElement(paragraph, q("fldSimple"), {q("instr"): " PAGE "})
        run = E.SubElement(field, q("r"))
        rpr = E.SubElement(run, q("rPr"))
        E.SubElement(rpr, q("rStyle"), {q("val"): "PageNumber"})
        E.SubElement(run, q("t")).text = "1"
        package[name] = E.tostring(footer, encoding="UTF-8", xml_declaration=True, standalone=True)
    with ZipFile(path, "w") as z:
        for name, data in package.items():
            z.writestr(name, data)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(docx: Path, pdf: Path, application: Path) -> dict:
    pdfinfo = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    pdftext = subprocess.run(["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", pdfinfo, re.M).group(1))
    page_size = re.search(r"^Page size:\s+(.+)$", pdfinfo, re.M).group(1).strip()
    with ZipFile(docx) as z:
        doc_xml = z.read("word/document.xml")
        E.fromstring(doc_xml)
        omml = doc_xml.count(b"<m:oMath")
        image_count = sum(1 for item in z.namelist() if item.startswith("word/media/"))
    claim_count = len(re.findall(r"(?m)^\*\*\d+、\*\*", application.read_text(encoding="utf-8")))
    with tempfile.TemporaryDirectory(prefix="lite3_patent_render_check_") as render_dir:
        subprocess.run(
            ["pdftoppm", "-f", "2", "-l", "2", "-r", "72", "-png", str(pdf), str(Path(render_dir) / "page")],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        figure_page = Image.open(next(Path(render_dir).glob("page-*.png"))).convert("RGB")
        central = figure_page.crop((35, 135, figure_page.width - 35, int(figure_page.height * 0.72)))
        visible_figure_pixels = sum(1 for pixel in central.getdata() if min(pixel) < 225)
    return {
        "reference_template": str(REFERENCE),
        "source": str(SOURCE),
        "application_source": str(application),
        "claims": claim_count,
        "pdf": {"path": str(pdf.resolve()), "bytes": pdf.stat().st_size, "sha256": sha256(pdf), "pages": pages, "page_size": page_size},
        "docx": {"path": str(docx.resolve()), "bytes": docx.stat().st_size, "sha256": sha256(docx), "omml_objects": omml, "embedded_images": image_count},
        "checks": {
            "pdf_readable": bool(pdftext.strip()) and "说明书摘要" in pdftext and "权利要求书" in pdftext and "技术领域" in pdftext,
            "docx_readable": True,
            "has_three_drawings": image_count >= 3 and visible_figure_pixels > 5000,
            "abstract_drawing_visible": visible_figure_pixels > 5000,
            "all_claims_present": claim_count == 15 and bool(re.search(r"(?m)^15、", pdftext)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the patent Word document")
    parser.add_argument("--with-pdf", action="store_true", help="also overwrite the deliverable PDF and validation record")
    options = parser.parse_args()
    pandoc = shutil.which("pandoc") or "/home/tjx/anaconda3/bin/pandoc"
    markdown = SOURCE.read_text(encoding="utf-8")
    generate_figures()
    application = ROOT / "application_reference_format.md"
    application.write_text(build_application_source(markdown), encoding="utf-8")
    reference_docx = convert_reference_doc(ROOT)
    docx = ROOT / (STEM + ".docx")
    pdf = ROOT / (STEM + ".pdf")
    subprocess.run(
        [
            pandoc,
            application.name,
            "--from=markdown+tex_math_dollars+raw_attribute",
            "--resource-path=.",
            "--reference-doc=" + str(reference_docx),
            "--output=" + str(docx),
        ],
        cwd=ROOT,
        check=True,
    )
    style_docx(docx)
    if options.with_pdf:
        with tempfile.TemporaryDirectory(prefix="lite3_patent_pdf_") as profile:
            subprocess.run(
                [
                    "libreoffice",
                    "-env:UserInstallation=file://" + profile,
                    "--headless",
                    "--convert-to",
                    "pdf:writer_pdf_Export",
                    "--outdir",
                    str(ROOT),
                    str(docx),
                ],
                cwd=ROOT,
                check=True,
                timeout=120,
            )
        (ROOT / "validation_reference_format.json").write_text(
            json.dumps(validate(docx, pdf, application), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        with tempfile.TemporaryDirectory(prefix="lite3_patent_docx_check_") as temp_dir:
            profile = Path(temp_dir) / "profile"
            rendered = Path(temp_dir) / "rendered"
            rendered.mkdir()
            subprocess.run(
                [
                    "libreoffice",
                    "-env:UserInstallation=file://" + str(profile),
                    "--headless",
                    "--convert-to",
                    "pdf:writer_pdf_Export",
                    "--outdir",
                    str(rendered),
                    str(docx),
                ],
                cwd=ROOT,
                check=True,
                timeout=120,
                stdout=subprocess.DEVNULL,
            )
            result = validate(docx, rendered / (docx.stem + ".pdf"), application)
            print(json.dumps(result["checks"], ensure_ascii=False))
    print(docx)


if __name__ == "__main__":
    main()
