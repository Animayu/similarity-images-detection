import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from imagededup.methods import CNN, PHash
from imagededup.utils import plot_duplicates
import pickle
from datetime import datetime
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity

# 设置页面配置
st.set_page_config(
    page_title="图片相似度欺诈检测系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和介绍
st.title("图片相似度欺诈检测系统")
st.markdown("""
本系统用于检测与输入图像相似度高于阈值的图像，辅助防欺诈检测。支持CNN和pHash两种算法。
- **CNN**：基于深度学习特征，适合语义级相似性检测
- **pHash**：基于感知哈希，适合轻微变化图片的快速检测
""")

# 侧边栏 - 参数设置
st.sidebar.header("参数设置")

# 选择算法
algorithm = st.sidebar.selectbox(
    "选择算法",
    ["CNN", "pHash", "两种方法都使用"]
)

# 设置阈值
if algorithm == "CNN" or algorithm == "两种方法都使用":
    cnn_threshold = st.sidebar.slider(
        "CNN相似度阈值 (越高要求越相似)", 
        min_value=0.5, 
        max_value=1.0, 
        value=0.85,
        step=0.01
    )

if algorithm == "pHash" or algorithm == "两种方法都使用":
    phash_threshold = st.sidebar.slider(
        "pHash距离阈值 (越低要求越相似)", 
        min_value=0, 
        max_value=20, 
        value=8,
        step=1
    )

# 设置显示的最相似图片数量
top_n = st.sidebar.slider(
    "显示最相似的图片数量", 
    min_value=1, 
    max_value=10, 
    value=5
)

# 设置图片库路径
image_library_path = st.sidebar.text_input(
    "图片库路径",
    value="./input/test_images/"
)

# 结果导出设置
export_results = st.sidebar.checkbox("导出结果", value=True)

# 数据加载和处理函数
@st.cache_data
def load_image_files(image_dir):
    """加载图片库中的所有图片文件路径"""
    if not os.path.exists(image_dir):
        return []
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return image_files

# 检查并创建特征缓存目录
cache_dir = "./cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 特征编码和缓存函数
@st.cache_data
def get_cnn_encodings(image_dir):
    """获取所有图片的CNN编码，优先使用缓存"""
    cache_file = os.path.join(cache_dir, "cnn_encodings.pkl")
    
    # 检查缓存是否存在且有效
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            encodings = pickle.load(f)
        st.sidebar.success(f"已加载CNN编码缓存，包含{len(encodings)}张图片")
        return encodings
    
    # 没有缓存，执行编码
    with st.spinner("正在进行CNN编码，首次运行可能需要几分钟..."):
        cnn = CNN()
        encodings = cnn.encode_images(image_dir=image_dir)
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(encodings, f)
        
        st.sidebar.success(f"CNN编码完成，已缓存{len(encodings)}张图片的特征")
        return encodings

@st.cache_data
def get_phash_encodings(image_dir):
    """获取所有图片的pHash编码，优先使用缓存"""
    cache_file = os.path.join(cache_dir, "phash_encodings.pkl")
    
    # 检查缓存是否存在且有效
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            encodings = pickle.load(f)
        st.sidebar.success(f"已加载pHash编码缓存，包含{len(encodings)}张图片")
        return encodings
    
    # 没有缓存，执行编码
    with st.spinner("正在进行pHash编码，首次运行可能需要几分钟..."):
        phasher = PHash()
        encodings = phasher.encode_images(image_dir=image_dir)
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(encodings, f)
        
        st.sidebar.success(f"pHash编码完成，已缓存{len(encodings)}张图片的特征")
        return encodings

# 图片编码函数
def encode_uploaded_image(img, method="cnn"):
    """对上传的图片进行编码"""
    if method == "cnn":
        # 保存临时文件
        temp_img_path = os.path.join(cache_dir, "temp_upload.jpg")
        img.save(temp_img_path)
        
        cnn = CNN()
        # 编码单个图片
        encoding = cnn.encode_image(image_file=temp_img_path)
        return encoding
    
    elif method == "phash":
        # 保存临时文件
        temp_img_path = os.path.join(cache_dir, "temp_upload.jpg")
        img.save(temp_img_path)
        
        phasher = PHash()
        # 编码单个图片
        encoding = phasher.encode_image(image_file=temp_img_path)
        return encoding
    
    return None

# 结果可视化函数
def visualize_results(uploaded_img, similar_images, scores, image_dir, method_name):
    """可视化相似图片结果"""
    # 显示原始上传图片
    st.subheader(f"{method_name}方法检测结果")
    
    # 如果没有相似图片
    if not similar_images:
        st.warning(f"未发现相似图片 (使用{method_name}方法)")
        return
    
    # 创建结果表格
    results_data = []
    
    # 使用列布局显示图片
    cols = st.columns(len(similar_images) + 1)
    
    # 第一列显示上传的图片
    with cols[0]:
        st.image(uploaded_img, caption="上传图片", use_column_width=True)
    
    # 其余列显示相似图片
    for i, (img_name, score) in enumerate(zip(similar_images, scores)):
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            with cols[i+1]:
                img = Image.open(img_path)
                st.image(img, caption=f"相似度: {score:.3f}", use_column_width=True)
                st.markdown(f"**文件名**: {img_name}")
                
                # 收集结果数据
                results_data.append({
                    "检测方法": method_name,
                    "相似图片": img_name,
                    "相似度分数": score
                })
    
    return pd.DataFrame(results_data)

# 导出结果函数
def get_download_link(df, filename="results.csv"):
    """生成下载链接"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">下载 {filename}</a>'
    return href

def compute_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    vec1_reshaped = np.array(vec1).reshape(1, -1)
    vec2_reshaped = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1_reshaped, vec2_reshaped)[0][0]

# 主要应用逻辑
def main():
    # 检查图片库路径
    if not os.path.exists(image_library_path):
        st.error(f"图片库路径 {image_library_path} 不存在！请检查路径设置。")
        return
    
    # 加载图片文件
    image_files = load_image_files(image_library_path)
    if not image_files:
        st.warning(f"图片库 {image_library_path} 中没有发现图片文件。")
        return
    
    st.info(f"已加载图片库，包含 {len(image_files)} 张图片。")
    
    # 文件上传部分
    st.header("上传图片进行相似性检测")
    
    uploaded_file = st.file_uploader("选择图片上传", type=["jpg", "jpeg", "png"])
    
    results_dfs = []  # 存储各方法的结果
    
    if uploaded_file is not None:
        # 显示进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 读取上传的图片
        img = Image.open(uploaded_file)
        st.image(img, caption="上传的图片", width=400)
        
        # 根据选择的算法执行相似性检测
        if algorithm in ["CNN", "两种方法都使用"]:
            # 更新进度
            status_text.text("准备CNN编码...")
            progress_bar.progress(10)
            
            # 获取库中所有图片的CNN编码
            cnn_encodings = get_cnn_encodings(image_library_path)
            
            # 更新进度
            status_text.text("对上传图片进行CNN编码...")
            progress_bar.progress(30)
            
            # 编码上传的图片
            uploaded_encoding = encode_uploaded_image(img, method="cnn")
            
            # 更新进度
            status_text.text("计算CNN相似度...")
            progress_bar.progress(50)
            
            # 计算与所有图片的相似度
            cnn_scores = {}
            for img_name, encoding in cnn_encodings.items():
                if encoding is not None and uploaded_encoding is not None:
                    similarity = compute_cosine_similarity(uploaded_encoding, encoding)
                    cnn_scores[img_name] = similarity
            
            # 根据相似度排序
            sorted_cnn = sorted(cnn_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 筛选高于阈值的结果
            filtered_cnn = [item for item in sorted_cnn if item[1] >= cnn_threshold]
            
            # 取前N个结果
            top_cnn = filtered_cnn[:top_n]
            
            # 更新进度
            status_text.text("CNN分析完成")
            progress_bar.progress(70)
            
            # 可视化结果
            if top_cnn:
                similar_images = [item[0] for item in top_cnn]
                scores = [item[1] for item in top_cnn]
                cnn_df = visualize_results(img, similar_images, scores, image_library_path, "CNN")
                if cnn_df is not None:
                    results_dfs.append(cnn_df)
            else:
                st.warning("未发现相似度高于阈值的图片 (CNN方法)")
        
        if algorithm in ["pHash", "两种方法都使用"]:
            # 更新进度
            status_text.text("准备pHash编码...")
            progress_bar.progress(75)
            
            # 获取库中所有图片的pHash编码
            phash_encodings = get_phash_encodings(image_library_path)
            
            # 更新进度
            status_text.text("对上传图片进行pHash编码...")
            progress_bar.progress(85)
            
            # 编码上传的图片
            uploaded_phash = encode_uploaded_image(img, method="phash")
            
            # 更新进度
            status_text.text("计算pHash距离...")
            progress_bar.progress(90)
            
            # 初始化PHash对象用于计算距离
            phasher = PHash()
            
            # 计算与所有图片的汉明距离
            phash_distances = {}
            for img_name, encoding in phash_encodings.items():
                if encoding is not None and uploaded_phash is not None:
                    distance = phasher._hamming_distance(uploaded_phash, encoding)
                    phash_distances[img_name] = distance
            
            # 根据距离排序（越小越相似）
            sorted_phash = sorted(phash_distances.items(), key=lambda x: x[1])
            
            # 筛选低于阈值的结果
            filtered_phash = [item for item in sorted_phash if item[1] <= phash_threshold]
            
            # 取前N个结果
            top_phash = filtered_phash[:top_n]
            
            # 更新进度
            status_text.text("pHash分析完成")
            progress_bar.progress(100)
            
            # 可视化结果
            if top_phash:
                similar_images = [item[0] for item in top_phash]
                # 对于pHash，我们用1.0减去归一化的距离作为"相似度分数"
                # 汉明距离范围是[0, 64]，所以我们用(64-distance)/64作为相似度
                scores = [(64 - item[1]) / 64 for item in top_phash]
                phash_df = visualize_results(img, similar_images, scores, image_library_path, "pHash")
                if phash_df is not None:
                    results_dfs.append(phash_df)
            else:
                st.warning("未发现相似度高于阈值的图片 (pHash方法)")
        
        # 合并所有结果并导出
        if results_dfs and export_results:
            all_results = pd.concat(results_dfs)
            st.header("导出检测结果")
            
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"相似图片检测结果_{timestamp}.csv"
            
            st.markdown(get_download_link(all_results, filename=filename), unsafe_allow_html=True)
            
            # 显示结果表格
            st.subheader("检测结果摘要")
            st.dataframe(all_results)

    # 页面底部水印
    st.markdown(
        """
        <div style='text-align: right; color: #888888; font-size: 12px; margin-top: 40px;'>
            © 数据中台部 · 数据挖掘组
        </div>
        """,
        unsafe_allow_html=True
    )

# 运行应用
if __name__ == "__main__":
    main() 