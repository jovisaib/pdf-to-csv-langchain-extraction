from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PDFPlumberLoader
from typing import List
import csv


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class StructuredExtraction(BaseModel):
    bill_to: str = Field(description="bill to")


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

pdf_reader = PDFPlumberLoader("./input2.pdf")
pages = pdf_reader.load_and_split()
text = " ".join(list(map(lambda page: page.page_content, [pages[8]])))


loan_summary_financial = """
    You have the following raw data (inside <context> tag) that is contained in a table:
    <context>
    {context}
    </context>
    
    The output must contain the following columns:
    - "Section Name" -> It starts with a letter (a, b, c, etc.) ([ ] or [X]) <name of the section>, include the name of the section.
    - "Section boolean" -> The boolean value of the section, if it has [X] then it is true, otherwise if [ ] it is false.
    - "All Contributions"
    - "Elective Deferrals"
    - "Matching"
    - "Nonelective"



    The values are boolean, if it has [X] then it is true, otherwise if [ ] it is false.
    Make sure values and columns match, so they don't break the CSV.
    Do NOT include any explanations, extra comments or markers like "```csv", only provide a CSV (.CSV format) directly compliant response.
"""



template = ChatPromptTemplate.from_template(loan_summary_financial)
chain = template | model | StrOutputParser()
res = chain.invoke({"context": text})
print(res)


# structured_prompt = model.with_structured_output(StructuredExtraction)
# result = structured_prompt.invoke(text)
# print(result)

# csv_filename = "invoice_data_output.csv"
# write_to_csv(result, csv_filename)
# print(f"CSV file '{csv_filename}' has been created with the structured data.")