from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PDFPlumberLoader
from typing import List
import csv

class StructuredExtraction(BaseModel):
    bill_to: str = Field(description="bill to")
    ship_to: str = Field(description="ship to")
    invoice_n_and_date: str = Field(description="Invoice No. & Date")
    sales_n_and_date: str = Field(description="Sales Order No. & Date")
    customer_vat_id: str = Field(description="Customer VAT ID No.")
    goods: List[str] = Field(description="List of Goods")
    quantity: List[int] = Field(description="Quantities for each good.")
    currency: List[str] = Field(description="Currency of each good")
    unit_price: List[float] = Field(description="Unit Price for each good")
    totals: List[float] = Field(description="Subtotals for each good")


def write_to_csv(data: StructuredExtraction, filename: str):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Bill To', 'Ship To', 'Invoice No. & Date', 'Sales Order No. & Date', 'Customer VAT ID No.',
                         'Good', 'Quantity', 'Currency', 'Unit Price', 'Total'])
        
        for i in range(len(data.goods)):
            writer.writerow([
                data.bill_to,
                data.ship_to,
                data.invoice_n_and_date,
                data.sales_n_and_date,
                data.customer_vat_id,
                data.goods[i],
                data.quantity[i],
                data.currency[i],
                data.unit_price[i],
                data.totals[i]
            ])


model = ChatOpenAI(model_name="gpt-4o-2024-08-06")

pdf_reader = PDFPlumberLoader("./input.pdf")
pages = pdf_reader.load_and_split()
text = " ".join(list(map(lambda page: page.page_content, pages)))


structured_prompt = model.with_structured_output(StructuredExtraction)
result = structured_prompt.invoke(text)
print(result)

csv_filename = "invoice_data_output.csv"
write_to_csv(result, csv_filename)
print(f"CSV file '{csv_filename}' has been created with the structured data.")