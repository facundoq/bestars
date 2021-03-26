
from pathlib import Path
from nbconvert import PDFExporter
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import datasets
output_folderpath = Path("reports/datasets_analysis")
output_folderpath.mkdir(parents=True,exist_ok=True)

for dataset_name in  datasets.datasets_by_name_all:
    print(f"Dataset {dataset_name}")
    print("     Reading notebook...")
    notebook_filename = "Exploratory analysis.ipynb"
    notebook = nbformat.read(notebook_filename,as_version=4)

    cell = notebook.cells[0]["source"] = f"dataset_name = '{dataset_name}'"

    # # 2. Instantiate the exporter. We use the `classic` template for now; we'll get into more details
    # # later about how to customize the exporter further.
    print("     Preprocessing notebook..")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(notebook)

    print("     Exporting to pdf..")
    exporter = PDFExporter()
    #exporter.template_name = 'classic'
    pdf_data, resources = exporter.from_notebook_node(notebook)

    output_filepath = output_folderpath / f"{dataset_name}.pdf"
    print("     Writing..")
    with open(output_filepath, "wb") as f:
        f.write(pdf_data)
        f.close()
    print("     Done")
