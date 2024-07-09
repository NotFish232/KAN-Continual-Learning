import streamlit as st

from utils.io import ExperimentReader, LogType, get_experiments


def main():
    for experiment in get_experiments():
        reader = ExperimentReader(experiment)

        st.write(f"### {experiment}")

        for entry in reader.read():
            if entry["type"] == LogType.graph:
                st.write(f"graph: {entry['name']}")
                st.pyplot(entry["data"])
                st.write("\n")


if __name__ == "__main__":
    main()
