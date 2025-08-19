import streamlit as st



# Create the navigation and run it
pg = st.navigation([st.Page('/content/pages/stream.py', icon="🔍",title="tracker"), st.Page('/content/pages/llm.py',icon="📈",title="Clarify your doubt with AI")])
pg.run()


 