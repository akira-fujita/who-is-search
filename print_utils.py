import streamlit as st


def debug_print(message, log_filename, debug_mode=False, use_streamlit=False):
    """
    共通デバッグ出力関数。
    
    Args:
        message (str): ログメッセージ。
        log_filename (str): ログファイル名。
        debug_mode (bool): デバッグモードが有効かどうか。
        use_streamlit (bool): Streamlit でメッセージを表示するか。
    """
    # ログをファイルに書き込む
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

    # 必要に応じて Streamlit またはコンソールに出力
    if debug_mode:
        if use_streamlit:
            st.write(message)
        else:
            print(message)
