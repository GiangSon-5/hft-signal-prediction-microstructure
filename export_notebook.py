"""Module trích xuất nội dung Notebook (.ipynb) ra file văn bản (.txt).

Module này đọc file Jupyter Notebook, phân tách các cell mã nguồn Python,
cell thuyết minh Markdown và kết quả đầu ra (Output logs/stream) thành
một file văn bản thuần phục vụ mục đích kiểm tra và tài liệu hóa.
"""

import json
from pathlib import Path


def export_notebook_to_txt(
    notebook_path: str = "notebooks/01_task1_signal_characterization.ipynb",
    output_txt_path: str = "reports/notebook_task1_export.txt",
) -> None:
    """Trích xuất toàn bộ Code, Markdown và Output từ file Notebook ra file .txt.

    Args:
        notebook_path (str): Đường dẫn đến file Jupyter Notebook (.ipynb).
        output_txt_path (str): Đường dẫn lưu file văn bản (.txt) đầu ra.
    """
    nb_file = Path(notebook_path)
    out_file = Path(output_txt_path)

    if not nb_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file notebook tại: {notebook_path}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(nb_file, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    lines = []
    lines.append("=" * 80)
    lines.append(f"TÀI LIỆU TRÍCH XUẤT TỪ NOTEBOOK: {nb_file.name}")
    lines.append("=" * 80)
    lines.append("")

    cells = nb_data.get("cells", [])

    for idx, cell in enumerate(cells, 1):
        cell_type = cell.get("cell_type", "unknown")
        lines.append(f"--- [CELL #{idx} - {cell_type.upper()}] ---")

        # 1. Trích xuất nội dung nguồn (Code hoặc Markdown)
        source = "".join(cell.get("source", []))
        lines.append(source)
        lines.append("")

        # 2. Nếu là cell CODE, trích xuất thêm Outputs
        if cell_type == "code":
            outputs = cell.get("outputs", [])
            if outputs:
                lines.append(">>> [OUTPUT LOGS]:")
                for out_idx, out in enumerate(outputs, 1):
                    output_type = out.get("output_type", "")
                    
                    # Stream output (print statements)
                    if output_type == "stream":
                        text = "".join(out.get("text", []))
                        lines.append(text.rstrip())
                    
                    # Execute result (biến xuất ở dòng cuối)
                    elif output_type in ["execute_result", "display_data"]:
                        data = out.get("data", {})
                        if "text/plain" in data:
                            plain_text = "".join(data["text/plain"])
                            lines.append(plain_text.rstrip())
                    
                    # Traceback error nếu có
                    elif output_type == "error":
                        ename = out.get("ename", "")
                        evalue = out.get("evalue", "")
                        lines.append(f"ERROR {ename}: {evalue}")
                lines.append("")

        lines.append("-" * 80)
        lines.append("")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Đã trích xuất thành công {len(cells)} cells ra file: {output_txt_path}")


if __name__ == "__main__":
    export_notebook_to_txt()
