from pathlib import Path
import shutil
import fitz
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'doc'
shutil.copyfile(ROOT / 'docs/visuals/iterative-rag-concept.png', OUT / 'iterative-rag-flow.png')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', fontName='Helvetica-Bold', fontSize=27, leading=32, textColor=colors.HexColor('#102844'), spaceAfter=15))
styles.add(ParagraphStyle(name='Deck', fontSize=12, leading=18, textColor=colors.HexColor('#476178'), spaceAfter=15))
styles['BodyText'].fontSize=10.5
styles['BodyText'].leading=13.5
styles['BodyText'].spaceAfter=7
styles['Heading2'].textColor=colors.HexColor('#087f8c')
styles['Heading2'].spaceBefore=13
styles['Heading2'].spaceAfter=8
styles.add(ParagraphStyle(name='Cell', fontSize=9, leading=12, spaceAfter=0))
styles.add(ParagraphStyle(name='CodeCustom', fontName='Courier', fontSize=8.5, leading=12, spaceAfter=12))
story=[]
def p(text, style='BodyText'): story.append(Paragraph(text, styles[style]))
def h(text): p(text,'Heading2')
def table(rows, widths):
    data=[[Paragraph(str(c),styles['Cell']) for c in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f3f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOX',(0,0),(-1,-1),.6,colors.HexColor('#cad8e2')),('INNERGRID',(0,0),(-1,-1),.4,colors.HexColor('#dce5ec')),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(t)

p('Recursive / Iterative<br/>Retrieval RAG','TitleCustom')
p('A practical guide to searching again when the first evidence is incomplete.','Deck')
h('1. Start with ordinary RAG')
p('<b>Retrieval-Augmented Generation (RAG)</b> combines document search with a language model. Documents are split into passages, converted into numerical vectors called embeddings, and indexed. A question is embedded too; the retriever selects relevant passages, and the model uses them to write an answer.')
p('A one-shot system retrieves once and then answers. This can miss a crucial rule when the question depends on facts spread across several passages. Similarity to a question does not mean a passage contains everything needed to answer it.')
h('2. Add a feedback loop')
p('<b>Iterative retrieval</b> repeats search when an evidence check finds missing information. Each search targets a specific gap, while useful evidence from earlier attempts is retained. The final answer still addresses the original question.')
story.append(Spacer(1,6))
story.append(Image(str(OUT/'iterative-rag-flow.png'),width=491,height=276.4))
p('Figure 1. The project flow: retrieve, accumulate, assess, and rewrite if needed. A retry limit or a repeated query stops further searching.','BodyText')

story.append(PageBreak())
p('How the retrieval loop works','TitleCustom')
p('The implementation in this repository is a bounded iterative workflow.','Deck')
table([
['Step','What happens and why'],
['1. Prepare the corpus','Split documents into chunks with stable IDs. Embed each chunk. This project keeps normalized vectors in memory.'],
['2. Search','Embed the current query, compute cosine similarity, and retrieve the top-k passages. Here, k means passages returned per search.'],
['3. Accumulate','Merge new passages into the evidence pool. Deduplicate by chunk ID so repeated hits do not create duplicate context.'],
['4. Assess','Ask the model whether the accumulated passages explicitly answer every part of the original question. Record missing facts and supporting IDs.'],
['5. Refine or stop','If evidence is sufficient, stop. Otherwise, rewrite the search toward missing facts, unless the retry budget is exhausted or the new query repeats.'],
['6. Generate','Use the accumulated passages to answer with chunk citations. If evidence remains incomplete, identify the unanswered parts.'],
],[95,396])
h('Iterative versus recursive: related, but different')
p('<b>Iterative retrieval</b> advances through successive attempts in a loop: search, inspect evidence, refine the query, and search again. Its state includes prior queries, gathered passages, and remaining retries.')
p('<b>Recursive retrieval</b> usually means that retrieval invokes further retrieval to resolve a dependency or expand a hierarchy. For example, a question is split into subquestions, each subquestion retrieves evidence, and their results are combined. Another pattern retrieves a summary first, then follows it to underlying document chunks. Terminology varies across systems.')
p('A recursive workflow needs a base case, such as a resolved subquestion, plus a depth or work limit. It may also use an iterative loop within each branch. Repeating a search alone does not imply recursive function calls.')
p('<b>In this project:</b> <font face="Courier">Pipeline.run()</font> in <font face="Courier">rag.py</font> uses a Python <font face="Courier">for</font> loop. It rewrites one query at a time; it does not call itself or build a tree of subquestions. The precise description is <b>iterative evidence gathering</b>.')

story.append(PageBreak())
p('A worked example','TitleCustom')
p('Illustrative retrieval sequence using the fictional Atlas handbook. Actual search order depends on the model and embeddings.','Deck')
p('<b>Question:</b> Can an Atlas employee who joined 20 days ago work remotely from Lisbon for 20 days, and who must approve it?')
table([
['Attempt','Evidence and decision'],
['First search','Suppose H001 is retrieved. It says Lisbon is approved, international remote work is limited to 30 days per year, and requests need manager and People Operations approval. But eligibility is governed by a separate probation policy.'],
['Gap check','The location, duration, and approvers are known. The missing fact is whether someone who joined 20 days ago is eligible. Answering now could incorrectly imply permission.'],
['Refined search','An illustrative rewrite is: "Atlas probation eligibility international remote work new employee 90 days". Suppose it retrieves H002: probation lasts 90 days, and employees are ineligible during probation.'],
['Combined answer','No. An employee who joined 20 days ago is still within the 90-day probation period and cannot use temporary international remote work [H002]. After probation, they may apply; Lisbon is approved, the annual limit is 30 days, and manager plus People Operations approval is required [H001].'],
],[95,396])
h('Bounds and stopping conditions in this repository')
p('With <b>max_retries = 3</b>, the pipeline performs at most <b>4 searches</b>: the initial search plus three retries. With <b>top_k = 1</b>, the evidence pool holds at most four unique passages, or fewer if results overlap. In general, unique context is bounded by <b>min(corpus size, (1 + max_retries) x top_k)</b>.')
p('The loop stops on sufficient evidence, an exhausted retry budget, or a repeated query (compared case-insensitively). A provider error ends the run with an error. The original question remains the reference for every assessment, reducing drift from query rewriting.')
h('Benefits, costs, and checks')
p('This approach can recover evidence that the initial search missed, especially for questions joining several policies or facts. Each extra round adds embedding and model calls, latency, and cost. It cannot recover a fact absent from the corpus, and an incorrect sufficiency judgment can stop the loop too early.')
p('Inspect evidence recall, answer support, missing facts, search count, and latency. Compare against a one-shot baseline with a comparable context budget: improvement may come partly from retrieving more passages. Valid citation IDs do not prove that every answer claim follows from the cited text.')
h('Project references')
p('<b>README.md:</b> workflow, retry semantics, and demonstration.<br/><b>rag.py:</b> Pipeline.retrieve(), Pipeline.run(), Euri.assess(), rewrite(), and generate().<br/><b>data/handbook.json:</b> fictional H001 and H002 policies.<br/><b>docs/visuals/iterative-rag-concept.png:</b> source image for Figure 1.')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#dce5ec')); canvas.line(52,43,543,43)
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#61768a'))
    canvas.drawString(52,29,'RETRACE  /  Retrieval RAG explained')
    canvas.drawRightString(543,29,str(doc.page))

path=OUT/'recursive_iterative_retrieval_rag.pdf'
doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=52,leftMargin=52,topMargin=44,bottomMargin=57,title='Recursive / Iterative Retrieval RAG',author='Retrace project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
pdf=fitz.open(path)
for i,page in enumerate(pdf):
    page.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(ROOT/f'tmp/pdfs/page-{i+1}.png')
print(f'Created {path} | {len(pdf)} pages')
for i,page in enumerate(pdf):
    print(f'Page {i+1}: {len(page.get_text())} text characters')
