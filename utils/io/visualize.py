import streamlit as st

from utils.io import ExperimentReader, get_experiment_plots, get_experiments


def main():
    for experiment in get_experiments():
        reader = ExperimentReader(experiment)
        reader.read()


        st.write(f"### {experiment}")

        st.write("## Graphs")
        plots = get_experiment_plots(experiment, reader.data)
        for name, plot in plots.items():
            st.write(f"Graph: {name}")
            st.plotly_chart(plot)

        st.write("## Data")
        for name, data in reader.data.items():
            st.write(f"{name}: {data.shape}")
            with st.expander("View Data"):
                st.write(str(data))

        
            
                


if __name__ == "__main__":
    main()
