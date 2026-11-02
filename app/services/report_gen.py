import os
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def clean_execution_logs(raw_log: str) -> str:
    """Pip installation logs aur raw base64 text ko logs block se hatata hai."""
    if not raw_log:
        return "# No execution logs available."

    lines = raw_log.splitlines()
    filtered_lines = []

    for line in lines:
        # 1. Base64 images and Data URLs ko strip karein
        if "IMAGE_BASE64:" in line or line.strip().startswith("data:image"):
            continue

        # 2. Pip installation logs/spinners/progress bars ignore karein
        pip_patterns = [
            r"^Collecting\s+",
            r"^Downloading\s+",
            r"^Requirement already satisfied:",
            r"^Installing collected packages:",
            r"^Successfully installed",
            r"^\s*[\-\|\\\/]\s*",  # Spinner animations
            r"^\s*\[\s*\d+\/\d+\s*\]",  # Progress indicators e.g., [1/3]
            r"^\x1b",  # Terminal color control codes
            r"━+",  # Progress bar lines
        ]

        if any(re.search(pattern, line.strip()) for pattern in pip_patterns):
            continue

        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines).strip()
    return cleaned if cleaned else "# No execution logs available."


class PDFReportGenerator:
    """Service to render HTML templates into downloadable executive PDF reports."""

    def __init__(self, template_dir: str = "app/templates"):
        self.template_path = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_path))
        )

    def generate_pdf(self, context: dict, output_pdf_path: str) -> str:
        """Renders context into HTML and exports to downloadable report."""
        template_name = "report.html"
        if not (self.template_path / template_name).exists():
            template_name = "report_template.html"

        # Logs clean-up pipeline trigger before Jinja rendering
        if "execution_output" in context:
            context["execution_output"] = clean_execution_logs(
                context["execution_output"]
            )

        template = self.env.get_template(template_name)
        html_content = template.render(**context)

        # 1. Primary Engine: WeasyPrint (if installed)
        try:
            import weasyprint

            weasyprint.HTML(string=html_content).write_pdf(output_pdf_path)
            return output_pdf_path
        except Exception:
            pass

        # 2. Native Sidecar HTML Fallback
        html_path = str(output_pdf_path).replace(".pdf", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path