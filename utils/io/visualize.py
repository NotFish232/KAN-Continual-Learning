import streamlit as st

from utils.io import ExperimentReader, LogType, get_experiments


def main():
    for experiment in get_experiments():
        reader = ExperimentReader(experiment)

        st.write(f"### {experiment}")

        for entry in reader.read():
            name, type, data = entry["name"], entry["type"], entry["data"]

            if type == LogType.graph:
                st.write(f"graph: {name}")
                st.plotly_chart(data)
                st.write("\n")
            
            if type == LogType.data:
                st.write(f"graph: {name}")
                st.write(data.shape)
                with st.expander("View Data"):
                    st.write(str(data))
                st.write("\n")
            

            


if __name__ == "__main__":
    main()
