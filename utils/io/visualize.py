import streamlit as st

from utils.io import ExperimentReader, LogType, get_experiments


def main():
    for experiment in get_experiments():
        reader = ExperimentReader(experiment)

        for entry in reader.read():
            if entry["type"] == LogType.graph:
                st.pyplot(entry["data"])


if __name__ == "__main__":
    main()
