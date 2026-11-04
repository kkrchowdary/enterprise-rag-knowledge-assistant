from pathlib import Path
import pymupdf


def extract_pdf_pages(file_path: str) -> list[dict]:

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    pages = []

    try:
        with pymupdf.open(path) as document:

            if document.page_count == 0:
                raise ValueError("PDF contains no pages.")

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = page.get_text(
                    "text",
                    sort=True
                ).strip()

                if text:
                    pages.append({
                        "source": path.name,
                        "page": page_number,
                        "text": text
                    })

    except pymupdf.FileDataError as error:
        raise ValueError(
            "Invalid or corrupted PDF document."
        ) from error

    if not pages:
        raise ValueError(
            "No extractable text found in PDF. "
            "The document may be scanned and require OCR."
        )

    return pages


if __name__ == "__main__":

    result = extract_pdf_pages(
        "data/documents/sample1.pdf"
    )

    for page in result:

        print(
            f"Page {page['page']}: "
            f"{len(page['text'])} characters"
        )

        print(page["text"][:300])