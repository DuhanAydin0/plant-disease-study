import streamlit as st
import requests

API_URL_DEFAULT = "http://127.0.0.1:5000/predict"

st.set_page_config(page_title="Plant Disease Demo", layout="centered")
st.title("Plant Disease Classification Demo")

st.markdown("Upload a leaf image, choose backend, and get prediction from Flask API.")

api_url = st.text_input("Flask API URL", value=API_URL_DEFAULT)

backend = st.selectbox(
    "Backend",
    options=[
        "model1_model2",
        "global_cnn",
        "cnn_svm",
        "transfer_learning",
    ],
    index=0
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


def render_result(out: dict):

    st.subheader("Prediction")

    # -------------------------------------------------
    # Case 1: model1_model2 style output
    # -------------------------------------------------
    if "plant" in out or "disease" in out or "ood" in out:
        st.write(f"**Plant:** {out.get('plant')}")
        st.write(f"**Disease:** {out.get('disease')}")
        st.write(f"**OOD:** {out.get('ood')}")
        return

    # -------------------------------------------------
    # Case 2: transfer_learning / global / cnn_svm
    # -------------------------------------------------
    top1 = out.get("top1")
    preds = out.get("preds")

    if isinstance(top1, dict):
        st.write(f"**Top-1:** {top1.get('class_name')}")
        st.write(f"**Probability:** {top1.get('prob'):.4f}")
    elif isinstance(preds, list) and len(preds) > 0:
        st.write(f"**Top-1:** {preds[0].get('class_name')}")
        st.write(f"**Probability:** {preds[0].get('prob'):.4f}")
    else:
        st.warning("Unknown output format. See raw JSON below.")


if uploaded is not None:

    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):

        try:
            files = {
                "image": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type
                )
            }

            data = {"backend": backend}

            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=120
            )

            if response.status_code != 200:
                st.error(f"API error ({response.status_code}): {response.text}")
            else:
                out = response.json()
                render_result(out)

                st.subheader("Raw JSON")
                st.json(out)

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")