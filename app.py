import requests
import json
import gradio as gr
# from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import pandas as pd
import time
from cnocr import CnOcr
# import pyocr as pcr
from sentence_transformers import SentenceTransformer, models, utilfrom docx2pdf import convert
word_embedding_model = models.Transformer('uer/sbert-base-chinese-nli', do_lower_case=True)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
ocr = CnOcr()
chat_url = 'https://Raghav001-my-api.hf.space/chatpdf'
headers = {
    'Content-Type': 'application/json',
}
# thread_pool_executor = ThreadPoolExecutor(max_workers=4)
history_max_len = 500
all_max_len = 3000


def get_emb(text):
    emb_url = 'https://souljoy-my-api.hf.space/embeddings'
    data = {"content": text}
    try:
        result = requests.post(url=emb_url,
                               data=json.dumps(data),
                               headers=headers
                               )
        return result.json()['data'][0]['embedding']
    except Exception as e:
        print('data', data, 'result json', result.json())


def doc_emb(doc: str):
    texts = doc.split('\n')
    # futures = []
    emb_list = embedder.encode(texts)
    # for text in texts:
    #     futures.append(thread_pool_executor.submit(get_emb, text))
    # for f in futures:
    #     emb_list.append(f.result())
    print('\n'.join(texts))
    return texts, emb_list, gr.Textbox.update(visible=True), gr.Button.update(visible=True), gr.Markdown.update(
        value="""success ! Let's talk"""), gr.Chatbot.update(visible=True)


def get_response(msg, bot, doc_text_list, doc_embeddings):
    # future = thread_pool_executor.submit(get_emb, msg)
    now_len = len(msg)
    req_json = {'question': msg}
    his_bg = -1
    for i in range(len(bot) - 1, -1, -1):
        if now_len + len(bot[i][0]) + len(bot[i][1]) > history_max_len:
            break
        now_len += len(bot[i][0]) + len(bot[i][1])
        his_bg = i
    req_json['history'] = [] if his_bg == -1 else bot[his_bg:]
    # query_embedding = future.result()
    query_embedding = embedder.encode([msg])
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    score_index = [[score, index] for score, index in zip(cos_scores, [i for i in range(len(cos_scores))])]
    score_index.sort(key=lambda x: x[0], reverse=True)
    print('score_index:\n', score_index)
    index_set, sub_doc_list = set(), []
    for s_i in score_index:
        doc = doc_text_list[s_i[1]]
        if now_len + len(doc) > all_max_len:
            break
        index_set.add(s_i[1])
        now_len += len(doc)
       # Maybe the paragraph is truncated wrong, so add the upper and lower paragraphs
        if s_i[1] > 0 and s_i[1] -1 not in index_set:
            doc = doc_text_list[s_i[1]-1]
            if now_len + len(doc) > all_max_len:
                break
            index_set.add(s_i[1]-1)
            now_len += len(doc)
        if s_i[1] + 1 < len(doc_text_list) and s_i[1] + 1 not in index_set:
            doc = doc_text_list[s_i[1]+1]
            if now_len + len(doc) > all_max_len:
                break
            index_set.add(s_i[1]+1)
            now_len += len(doc)

    index_list = list(index_set)
    index_list.sort()
    for i in index_list:
        sub_doc_list.append(doc_text_list[i])
    req_json['doc'] = '' if len(sub_doc_list) == 0 else '\n'.join(sub_doc_list)
    data = {"content": json.dumps(req_json)}
    print('data:\n', req_json)
    result = requests.post(url=chat_url,
                           data=json.dumps(data),
                           headers=headers
                           )
    res = result.json()['content']
    bot.append([msg, res])
    return bot[max(0, len(bot) - 3):]


def up_file(files):
    doc_text_list = []
    for idx, file in enumerate(files):
        print(file.name)
        with pdfplumber.open(file.name) as pdf:
            for i in range(len(pdf.pages)):
                # Read page i+1 of a PDF document
                page = pdf.pages[i]
                res_list = page.extract_text().split('\n')[:-1]

                for j in range(len(page.images)):
                   # Get the binary stream of the image
                    img = page.images[j]
                    file_name = '{}-{}-{}.png'.format(str(time.time()), str(i), str(j))
                    with open(file_name, mode='wb') as f:
                        f.write(img['stream'].get_data())
                    try:
                        res = ocr.ocr(file_name)
                    except Exception as e:
                        res = []
                    if len(res) > 0:
                        res_list.append(' '.join([re['text'] for re in res]))

                tables = page.extract_tables()
                for table in tables:
                    # The first column is used as the header
                    df = pd.DataFrame(table[1:], columns=table[0])
                    try:
                        records = json.loads(df.to_json(orient="records", force_ascii=False))
                        for rec in records:
                            res_list.append(json.dumps(rec, ensure_ascii=False))
                    except Exception as e:
                        res_list.append(str(df))

                doc_text_list += res_list
    doc_text_list = [str(text).strip() for text in doc_text_list if len(str(text).strip()) > 0]
    print(doc_text_list)
    return gr.Textbox.update(value='\n'.join(doc_text_list), visible=True), gr.Button.update(
        visible=True), gr.Markdown.update(
        value="Processing")


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file = gr.File(file_types=['.pdf'], label='Click to upload Document', file_count='multiple')
            doc_bu = gr.Button(value='Submit', visible=False)
            txt = gr.Textbox(label='result', visible=False)
            doc_text_state = gr.State([])
            doc_emb_state = gr.State([])
        with gr.Column():
            md = gr.Markdown("Please Upload the PDF")
            chat_bot = gr.Chatbot(visible=False)
            msg_txt = gr.Textbox(label='Ask Questions', placeholder='write', visible=False)
            chat_bu = gr.Button(value='Proceed', visible=False)

    file.change(up_file, [file], [txt, doc_bu, md])
    doc_bu.click(doc_emb, [txt], [doc_text_state, doc_emb_state, msg_txt, chat_bu, md, chat_bot])
    chat_bu.click(get_response, [msg_txt, chat_bot, doc_text_state, doc_emb_state], [chat_bot])

if __name__ == "__main__":
    demo.queue().launch()
    # demo.queue().launch(share=False, server_name='172.22.2.54', server_port=9191)
