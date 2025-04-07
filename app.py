import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(CURRENT_DIR, "../../utils")
sys.path.append(os.path.normpath(UTILS_PATH))

from print_utils import debug_print
import requests
from bs4 import BeautifulSoup
 
import urllib3
import re
from urllib.parse import quote
from urllib.parse import urljoin
from dotenv import load_dotenv
import pdfplumber
from io import BytesIO
import csv
from datetime import datetime
import time
import streamlit as st
from io import StringIO

# .envファイルを読み込む
load_dotenv()

# SerpApiのAPIキーを環境変数から取得
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    raise ValueError("APIキーが設定されていません。環境変数 'SERPAPI_API_KEY' を設定してください。")

# InsecureRequestWarningを無効にする
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def call_search(keywords, num_results=5, log_filename="debug_log.txt", debug_mode=False, use_streamlit=False):
    """
    キーワードでGoogle検索を行い、取得した結果のURLリストを返す。
    
    Args:
        keywords (list): 検索キーワードのリスト。
        num_results (int): 取得する結果の最大件数。
        log_filename (str): ログファイルの名前。
        debug_mode (bool): デバッグモードが有効かどうか。
        use_streamlit (bool): Streamlit環境でのデバッグ出力を有効にするか。
    Returns:
        list: 検索結果から取得したURLのリスト。
    """
    st.write (log_filename)
    print ("------------------")
    # keywordsをスペースで結合してクエリにする
    query = " ".join(keywords)
    debug_print(f"[DEBUG - call_search] Query: {query}, Number of Results: {num_results}", log_filename, debug_mode, use_streamlit)

    params = {
        "engine": "google",
        "q": query,
        "hl": "ja",  # 日本語
        "gl": "JP",  # 地域を日本に設定
        "device": "desktop",  # デスクトップでの検索
        "num": num_results,
        "api_key": SERPAPI_API_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])

    # デバッグ用に検索結果の数を出力
    debug_print(f"[DEBUG - call_search] Number of organic results: {len(organic_results)}", log_filename, debug_mode, use_streamlit)

    # URLリストのみを返す
    urls = [result.get("link") for result in organic_results if result.get("link")]
    debug_print(f"[DEBUG - call_search] Extracted URLs: {urls}", log_filename, debug_mode, use_streamlit)
    return urls


def check_whois_registration_info(url) -> bool:
    """
    Check a JPRS WHOIS page for the presence of '登録担当者'.

    Args:
        key (str): The search keyword (e.g., a domain holder name).
        type (str): The search type (e.g., "DOM-HOLDER").

    Returns:
        bool: True if '登録担当者' is found on the page, False otherwise.
    """
    try:
        # Fetch the page content
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raise exception for HTTP errors (4xx/5xx)
    except requests.RequestException as e:
        # Handle any network/HTTP errors gracefully by returning False
        return False

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    # Get all text from the page
    page_text = soup.get_text()

    # Check if the target string is in the page text
    if "登録担当者" in page_text:
        return True
    else:
        return False


# ## 正規表現をつかった検索
def extract_emails_from_page(url, log_filename="debug_log.txt", debug_mode=False, use_streamlit=False):
    """
    指定されたURLのページまたはPDFファイルからメールアドレスを抽出する。

    Args:
        url (str): 対象のURL。
        log_filename (str): ログファイルの名前。
        debug_mode (bool): デバッグモードが有効かどうか。
        use_streamlit (bool): Streamlit環境でのデバッグ出力を有効にするか。

    Returns:
        list: 抽出したメールアドレスのリスト。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
    }
    debug_print(f"[DEBUG - extract_emails_from_page] Fetching URL: {url}", log_filename, debug_mode, use_streamlit)

    try:
        response = requests.get(url, headers=headers, verify=False)
        debug_print(f"[DEBUG - extract_emails_from_page] Response Status Code: {response.status_code}", log_filename, debug_mode, use_streamlit)

        response.raise_for_status()

        # メールアドレスを抽出する正規表現
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        emails = []

        # (1) PDF の場合はダウンロードして解析
        if url.lower().endswith(".pdf") or "application/pdf" in response.headers.get("Content-Type", ""):
            debug_print(f"[DEBUG - extract_emails_from_page] PDF detected: {url}", log_filename, debug_mode, use_streamlit)

            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                emails = re.findall(email_pattern, text)

        # (2) HTML の場合は通常の方法で解析
        else:
            soup = BeautifulSoup(response.text, "html.parser")

            # <a> タグの `href` からメールアドレスを抽出
            for link in soup.find_all("a", href=True):
                match = re.search(email_pattern, link["href"])
                if match:
                    emails.append(match.group(0))

            # <p> や <strong> タグのテキストから抽出
            for tag in soup.find_all(["p", "strong", "li", "div", "span"]):
                match = re.search(email_pattern, tag.get_text())
                if match:
                    emails.append(match.group(0))

            # ページ全体のテキストから抽出
            text = soup.get_text()
            emails.extend(re.findall(email_pattern, text))

        # 重複を削除
        emails = list(set(emails))

        debug_print(f"[DEBUG - extract_emails_from_page] Extracted Emails: {emails}", log_filename, debug_mode, use_streamlit)
        return emails
    except requests.RequestException as e:
        error_message = f"[ERROR - extract_emails_from_page] Failed to fetch {url}: {e}"
        debug_print(error_message, log_filename, debug_mode, use_streamlit)
        return []


def extract_contact_emails_from_1layer(url, log_filename="debug_log.txt", debug_mode=False, use_streamlit=False):
    """
    JPRS WHOISページのURLから、登録担当者と技術連絡担当者のメールアドレスを抽出する。

    Args:
        url (str): WHOISページのURL。
        log_filename (str): ログファイル名。
        debug_mode (bool): デバッグ出力の有無。
        use_streamlit (bool): Streamlitでの出力を使うかどうか。

    Returns:
        dict: {
            "registrant_contact_url": str,
            "tech_contact_url": str,
            "registrant_email": str,
            "tech_email": str
        }
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        debug_print(f"[ERROR - extract_contact_emails_from_whois] Failed to fetch WHOIS page: {e}", log_filename, debug_mode, use_streamlit)
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    contact_info = {
        "registrant_contact_url": None,
        "registrant_name": None,
        "registrant_email": None,
        "tech_contact_url": None,
        "tech_name": None,
        "tech_email": None
    }

    # 登録担当者と技術連絡担当者のリンクを取得
    pre_tag = soup.find("pre")
    if not pre_tag:
        debug_print("[ERROR - extract_contact_emails_from_whois] <pre> tag not found.", log_filename, debug_mode, use_streamlit)
        return contact_info

    links = pre_tag.find_all("a", href=True)
    for a in links:
        full_url = urljoin(url, a['href'].replace("&amp;", "&"))
        prev = a.previous_sibling
        if prev and isinstance(prev, str):
            if "m. [登録担当者]" in prev:
                contact_info["registrant_contact_url"] = full_url
                # print("registrant_contact_url:", full_url)
            elif "n. [技術連絡担当者]" in prev:
                contact_info["tech_contact_url"] = full_url
                # print("tech_contact_url:", full_url)

    # 各リンクからメールアドレスを取得
    def get_email_from_contact_page(contact_url, role):
        try:
            res = requests.get(contact_url, headers=headers, timeout=10)
            res.raise_for_status()
            contact_soup = BeautifulSoup(res.text, "html.parser")
            contact_pre = contact_soup.find("pre")
            if contact_pre:
                name_match = re.search(r"b\.\s*\[氏名\]\s*(.+)", contact_pre.text)
                if name_match:
                    contact_info[f"{role}_name"] = name_match.group(1)

                    match = re.search(r"d\. \[電子メイル\]\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", contact_pre.text)
                    if match:
                        return match.group(1)
        except requests.RequestException as e:
            debug_print(f"[ERROR] Failed to fetch contact page: {e}", log_filename, debug_mode, use_streamlit)
        return None

    if contact_info["registrant_contact_url"]:
        contact_info["registrant_email"] = get_email_from_contact_page(contact_info["registrant_contact_url"], "registrant")

    if contact_info["tech_contact_url"]:
        contact_info["tech_email"] = get_email_from_contact_page(contact_info["tech_contact_url"], "tech")

    # for debbug
    print("登録担当者_url:", contact_info["registrant_contact_url"])
    print(" 登録担当者_email:", contact_info["registrant_email"])
    print("技術連絡担当者_url:", contact_info["tech_contact_url"])
    print(" 技術連絡担当者_email:", contact_info["tech_email"])

    return contact_info


def extract_contact_emails_from_2layer(url, log_filename="debug_log.txt", debug_mode=False, use_streamlit=False, wait_seconds=0):
    """
    DOM-HOLDERタイプのページから複数のドメインリンクを辿ってEmailを取得する。
    
    Args:
        url (str): WHOISページのURL。
        log_filename (str): ログファイル名。
        debug_mode (bool): デバッグ出力の有無。
        use_streamlit (bool): Streamlitでの出力を使うかどうか。
    
    Returns:
        dict: {
            "domain_links": [str],
            "emails": [str]
        }
    """
    debug_print(f"[DEBUG] wait_seconds is set to {wait_seconds}", log_filename, debug_mode, use_streamlit)
    processed_count = 0
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        debug_print(f"[ERROR - extract_contact_emails_from_2layer] Failed to fetch WHOIS page: {e}", log_filename, debug_mode, use_streamlit)
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    pre_tags = soup.find_all("pre")
    if not pre_tags:
        debug_print("[ERROR - extract_contact_emails_from_2layer] <pre> tags not found.", log_filename, debug_mode, use_streamlit)
        return {}

    domain_links = []
    emails = []
    names = []
    emails_found_count = 0

    for pre_tag in pre_tags:
        # Split by lines to process each line separately
        lines = pre_tag.decode_contents().splitlines()
        for line in lines:
            soup_line = BeautifulSoup(line, "html.parser")
            a_tag = soup_line.find("a", href=True)
            if a_tag:
                full_url = urljoin(url, a_tag['href'].replace("&amp;", "&"))
                domain_links.append(full_url)
                processed_count += 1
                if processed_count > 0 and processed_count % 25 == 0:
                    debug_print(f"[INFO] Waiting for {wait_seconds} seconds after processing {processed_count} links...", log_filename, debug_mode, use_streamlit)
                    time.sleep(int(wait_seconds))
                try:
                    res = requests.get(full_url, headers=headers, timeout=10)
                    res.raise_for_status()
                    contact_soup = BeautifulSoup(res.text, "html.parser")
                    contact_pre = contact_soup.find("pre")
                    if contact_pre:
                        email_match = re.search(r"\[Email\]\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", contact_pre.text)
                        if email_match:
                            emails.append(email_match.group(1))
                            emails_found_count += 1
                        else:
                            emails.append(f"[No Email Found] {full_url}")
                        
                        name_match = re.search(r"\[名前\]\s+(.+)", contact_pre.text)
                        if name_match:
                            names.append(name_match.group(1))
                        else:
                            names.append("[No Name Found]")
                    else:
                        emails.append(f"[No Email Found] {full_url}")
                except requests.RequestException as e:
                    debug_print(f"[ERROR - extract_contact_emails_from_2layer] Failed to fetch contact page: {e}", log_filename, debug_mode, use_streamlit)
                    emails.append(f"[No Email Found] {full_url}")

    print("Total processed domain links:", processed_count)
    actual_emails = [e for e in emails if not e.startswith("[No Email Found]")]
    print("Total emails found:", len(actual_emails))
    return {
        "domain_links": domain_links,
        "emails": emails,
        "names": names,
        "processed_links_count": processed_count,
        "emails_found_count": len(actual_emails)
    }


def save_results_to_csv(results, filename=None):
    """
    検索結果のURLと抽出したメールアドレスをCSVに保存する。
    ファイル名に実行時間を追加し、メールがない場合は "NONE" を入力。
    カンマの後に半角スペースを挿入。

    Args:
        results (list of tuples): 各URLと対応するメールアドレスのリスト。
        filename (str, optional): 出力するCSVファイルの名前。デフォルトは None（実行時間を追加）。
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"email_results_{timestamp}.csv"

    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["URL", "Emails"])  # ヘッダー行

            for url, emails in results:
                if not emails:
                    emails = ["NONE"]  # メールアドレスがない場合
                row = [url] + emails
                writer.writerow([", ".join(row)])  # カンマの後にスペースを挿入

        print(f"結果を {filename} に保存しました。")
    except Exception as e:
        print(f"[ERROR] CSVファイルの保存中にエラーが発生しました: {e}")


def process_uploaded_csv_file(uploaded_file, wait_seconds, output_area, download_btn_placeholder):
    import pandas as pd
    csv_data = uploaded_file.read().decode("shift_jis")
    df = pd.read_csv(StringIO(csv_data), encoding="shift_jis", header=None, names=["検索タイプ", "検索ワード"])
    search_type_options = {
        "ドメイン名情報": "DOM",
        "ドメイン名情報(登録者)": "DOM-HOLDER",
        "ネームサーバーホスト情報": "HOST",
        "ネームサーバーホスト情報(IPv4)": "HOST-IP",
        "ネームサーバーホスト情報(IPv6)": "HOST-V6",
        "担当者情報": "POC"
    }
    all_results = []
    for index, row in df.iterrows():
        label = row["検索タイプ"]
        search_type_label_map = {
            "ドメイン名情報": "DOM",
            "ドメイン名情報(登録者)": "DOM-HOLDER",
            "ネームサーバーホスト情報": "HOST",
            "ネームサーバーホスト情報(IPv4)": "HOST-IP",
            "ネームサーバーホスト情報(IPv6)": "HOST-V6",
            "担当者情報": "POC"
        }
        
        search_type = search_type_label_map.get(label.strip(), label.strip())
        search_keyword = row["検索ワード"]
        if not search_type or not search_keyword:
            continue
        debug_print("[DEBUG - process_uploaded_csv_file] search_type, keyword", search_type, search_keyword)

        target_url = f"https://whois.jprs.jp/?key={quote(search_keyword)}&type={search_type}"
        output_area.write(f"▶ 処理対象: {search_type}, {search_keyword}")

        is_1layer = check_whois_registration_info(target_url)
        if is_1layer:
            result = extract_contact_emails_from_1layer(target_url, debug_mode=True, use_streamlit=True)
            result["search_type"] = search_type
            result["search_keyword"] = search_keyword
            all_results.append(result)
            if wait_seconds > 0:
                output_area.write(f"{wait_seconds}秒待機中...")
                time.sleep(wait_seconds)
        else:
            result = extract_contact_emails_from_2layer(target_url, debug_mode=True, use_streamlit=True, wait_seconds=wait_seconds)
            result["search_type"] = search_type
            result["search_keyword"] = search_keyword
            all_results.append(result)
            if wait_seconds > 0:
                output_area.write(f"{wait_seconds}秒待機中...")
                time.sleep(wait_seconds)
        output_area.write("取得結果:")
        # output_area.json(result)

    # ダウンロード用CSVに変換
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)

    for result in all_results:
        writer.writerow(["検索タイプ", result.get("search_type", "")])
        writer.writerow(["検索ワード", result.get("search_keyword", "")])
        writer.writerow([])

        if "registrant_contact_url" in result:
            writer.writerow(["種別", "URL", "Name", "Email"])
            # writer.writerow(["登録担当者", result.get("registrant_contact_url", ""), result.get("registrant_name", ""), result.get("registrant_email", "")])
            # writer.writerow(["技術連絡担当者", result.get("tech_contact_url", ""), result.get("tech_name", ""), result.get("tech_email", "")])
        else:
            writer.writerow(["URL", "Name", "Email"])
            for url, name, email in zip(result.get("domain_links", []), result.get("names", []), result.get("emails", [])):
                writer.writerow([url, name, email])
        writer.writerow([])

    # CSV出力
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{timestamp}.csv"
    csv_data = csv_buffer.getvalue().encode("shift_jis")
    download_btn_placeholder.download_button("CSVをダウンロード", csv_data, file_name=filename, mime="text/csv")
    output_area.text_area("CSV出力プレビュー", csv_data, height=300)
    return all_results


def main(key, type, log_filename="debug_log.txt", debug_mode=False, use_streamlit=False, wait_seconds=5):
    """
    メイン処理。キーワードを使用して検索し、結果を表示し、メールアドレスを抽出し、CSVに保存。

    Args:
        keywords (list): 検索キーワードのリスト。
        log_filename (str): ログファイルの名前。
        debug_mode (bool): デバッグモードが有効かどうか。
        use_streamlit (bool): Streamlit環境でのデバッグ出力を有効にするか。
    """
    debug_print("[DEBUG - main] Starting search process", log_filename, debug_mode, use_streamlit)
    # urls = call_search(keywords, num_results, log_filename, debug_mode, use_streamlit)

    # URL-encode the key to ensure it can be used in a URL
    encoded_key = quote(key, safe='')
    # Construct the WHOIS search URL
    url = f"https://whois.jprs.jp/?key={encoded_key}&type={type}"

    is1layer = check_whois_registration_info(url)
    print ("Target URL ", url)

    if is1layer == True:
        print ("1 layer type")
        ret = extract_contact_emails_from_1layer(url, log_filename, debug_mode, use_streamlit)
    else:
        print ("2 layer type")
        ret = extract_contact_emails_from_2layer(url, log_filename, debug_mode, use_streamlit, wait_seconds)
        print("取得したドメインリンク数:", len(ret.get("domain_links", [])))
        for link in ret.get("domain_links", []):
            print(" -", link)
        actual_emails = [e for e in ret.get("emails", []) if not e.startswith("[No Email Found]")]
        print("抽出されたメールアドレス数:", len(actual_emails))
        for email in actual_emails:
            print(" -", email)


def run_streamlit_ui():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>WHOIS Emailサーチ</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><a href='https://whois.jprs.jp/' target='_blank'>https://whois.jprs.jp/</a></p>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 2])
    with right_col:
        output_area = st.empty()
        download_btn_placeholder = st.empty()

    def generate_csv_download_button(result, placeholder, search_type=None, search_keyword=None):
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        # Add search metadata as the first row
        writer.writerow(["検索タイプ", search_type])
        writer.writerow(["検索ワード", search_keyword])
        writer.writerow([])  # Empty line for spacing
        
        if "registrant_contact_url" in result:
            # 1-layer result format
            writer.writerow(["種別", "URL", "Name", "Email"])
            writer.writerow([
                "登録担当者",
                result.get("registrant_contact_url", "[No URL]"),
                result.get("registrant_name", "[No Name]"),
                result.get("registrant_email", "[No Email]")
            ])
            writer.writerow([
                "技術連絡担当者",
                result.get("tech_contact_url", "[No URL]"),
                result.get("tech_name", "[No Name]"),
                result.get("tech_email", "[No Email]")
            ])
        else:
            # 2-layer result format
            writer.writerow(["URL", "Name", "Email"])
            for url, name, email in zip(result.get("domain_links", []), result.get("names", []), result.get("emails", [])):
                writer.writerow([url, name, email])
        
        csv_data = csv_buffer.getvalue()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{timestamp}.csv"
        placeholder.download_button("CSVをダウンロード", csv_data, file_name=filename, mime="text/csv")

    with left_col:
        st.header("検索設定")
        with st.form("main_form"):
            search_type_options = {
                "ドメイン名情報": "DOM",
                "ドメイン名情報(登録者)": "DOM-HOLDER",
                "ネームサーバーホスト情報": "HOST",
                "ネームサーバーホスト情報(IPv4)": "HOST-IP",
                "ネームサーバーホスト情報(IPv6)": "HOST-V6",
                "担当者情報": "POC"
            }

            selected_label = st.selectbox("検索タイプ", list(search_type_options.keys()))
            search_type = search_type_options[selected_label]
            search_keyword = st.text_input("検索ワード")
            uploaded_file = st.file_uploader("CSVファイルをアップロード（オプション）", type=["csv"])
            wait_seconds = st.slider("25件あたりの待ち時間（秒）", min_value=0, max_value=300, value=60, step=5)
            execute_button = st.form_submit_button("実行")
            
            if execute_button:
                if uploaded_file is not None:
                    all_results = process_uploaded_csv_file(uploaded_file, wait_seconds, output_area, download_btn_placeholder)
                    # Generate combined CSV for all results
                    combined_csv_buffer = StringIO()
                    writer = csv.writer(combined_csv_buffer)
                    for result in all_results:
                        writer.writerow(["検索タイプ", result.get("search_type", "")])
                        writer.writerow(["検索ワード", result.get("search_keyword", "")])
                        writer.writerow([])
                        if "registrant_contact_url" in result:
                            writer.writerow(["種別", "URL", "Name", "Email"])
                            writer.writerow(["登録担当者", result.get("registrant_contact_url", ""), result.get("registrant_name", ""), result.get("registrant_email", "")])
                            writer.writerow(["技術連絡担当者", result.get("tech_contact_url", ""), result.get("tech_name", ""), result.get("tech_email", "")])
                        else:
                            writer.writerow(["URL", "Name", "Email"])
                            for url, name, email in zip(result.get("domain_links", []), result.get("names", []), result.get("emails", [])):
                                writer.writerow([url, name, email])
                        writer.writerow([])
                    combined_csv_data = combined_csv_buffer.getvalue()
                    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    filename = f"combined_{timestamp}.csv"
                    download_btn_placeholder.download_button("全結果CSVをダウンロード", combined_csv_data, file_name=filename, mime="text/csv")
                elif search_keyword:
                    target_url = f"https://whois.jprs.jp/?key={quote(search_keyword)}&type={search_type}"
                    output_area.write(f"ターゲットURL: {target_url}")
                    is_1layer = check_whois_registration_info(target_url)
                    if is_1layer:
                        output_area.write("1レイヤーのページとして処理します。")
                        result = extract_contact_emails_from_1layer(target_url, debug_mode=True, use_streamlit=True)
                        output_area.json(result)
                        generate_csv_download_button(result, download_btn_placeholder, search_type, search_keyword)
                    else:
                        output_area.write(f"25件あたり　{wait_seconds}秒 の待ち時間が発生します。しばらくお待ちください")
                        result = extract_contact_emails_from_2layer(target_url, debug_mode=True, use_streamlit=True, wait_seconds=wait_seconds)
                        output_area.json(result)
                        generate_csv_download_button(result, download_btn_placeholder, search_type, search_keyword)
                else:
                    output_area.write("検索ワードを入力してください。")


if __name__ == "__main__":
    import __main__
    if hasattr(__main__, '__file__') and __main__.__file__.endswith("app.py"):
        run_streamlit_ui()
    else:
        print("このスクリプトは通常のPython実行では処理を開始しません。Streamlitで実行してください。")
