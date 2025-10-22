import numpy as np
import streamlit as st
import plotly.graph_objs as go

def kmc_diffusion_2D_question_II(num_steps, Gamma1, Gamma2, a=1.0, b=1.0):
    d1 = b
    d2 = a
    Gamma = Gamma1 + Gamma2
    directions_type1 = np.array([[d1, 0], [-d1, 0], [0, d1], [0, -d1]])
    directions_type2 = np.array([[d2, d2], [d2, -d2], [-d2, d2], [-d2, -d2]])
    positions = np.zeros((num_steps+1, 2))
    times = np.zeros(num_steps+1)
    prob_gamma1 = Gamma1 / Gamma
    for i in range(1, num_steps+1):
        r = np.random.rand()
        if r < prob_gamma1:
            step = directions_type1[np.random.choice(4)]
        else:
            step = directions_type2[np.random.choice(4)]
        positions[i] = positions[i-1] + step
        times[i] = times[i-1] + 1 / Gamma
    return positions, times

def compute_msd(positions):
    N = len(positions)
    max_t = N // 4
    msd = np.zeros(max_t)
    for t in range(1, max_t):
        diffs = positions[t:] - positions[:-t]
        squared_displacements = np.sum(diffs**2, axis=1)
        msd[t] = np.mean(squared_displacements)
    t_vals = np.arange(max_t)
    return t_vals[1:], msd[1:]

def estimate_diffusion_coefficient(t_vals, msd_vals):
    coeffs = np.polyfit(t_vals, msd_vals, 1)
    return coeffs[0] / 4

def analytical_diffusion_coefficient_II(Gamma1, Gamma2, a=1.0, b=1.0):
    return 0.25 * (Gamma1 * b**2 + Gamma2 * a**2)

def main():
    st.title("Modified KMC Diffusion (Question II)")
    st.set_page_config(layout="wide")
    a = st.number_input("Parameter a", value=1.0)
    b = st.number_input("Parameter b", value=1.0)
    num_steps = st.slider("Number of simulation steps", 1000, 50000, 10000, step=1000)
    Gamma1 = st.slider("Gamma1 frequency (transfer)", 0.0, 1.0, 0.5, step=0.05)
    Gamma2 = st.slider("Gamma2 frequency (90° rotation)", 0.0, 1.0, 0.5, step=0.05)
    if st.button("Run simulation"):
        traj, times = kmc_diffusion_2D_question_II(num_steps, Gamma1, Gamma2, a, b)
        t_vals, msd_vals = compute_msd(traj)
        D_num = estimate_diffusion_coefficient(t_vals, msd_vals)
        D_ana = analytical_diffusion_coefficient_II(Gamma1, Gamma2, a, b)
        st.session_state['modified_results'] = (traj, t_vals, msd_vals, D_num, D_ana)

    # Display persistent results if available
    if 'modified_results' in st.session_state:
        traj, t_vals, msd_vals, D_num, D_ana = st.session_state['modified_results']
        st.metric("Estimated numerical diffusion D", f"{D_num:.6f}")
        st.metric("Analytical diffusion D", f"{D_ana:.6f}")
        x_vals, y_vals = traj[:,0], traj[:,1]
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', line=dict(color='lightgray'), marker=dict(size=3), name='Trajectory'))
        fig_traj.add_trace(go.Scatter(x=[x_vals[0]], y=[y_vals[0]], mode='markers', marker=dict(color='green', size=10), name='Start'))
        fig_traj.add_trace(go.Scatter(x=[x_vals[-1]], y=[y_vals[-1]], mode='markers', marker=dict(color='red', size=10), name='End'))
        fig_traj.update_layout(title="Trajectory (question II)", xaxis_title="x", yaxis_title="y", yaxis_scaleanchor="x")
        st.plotly_chart(fig_traj, use_container_width=True)
        fig_msd = go.Figure()
        fig_msd.add_trace(go.Scatter(x=t_vals, y=msd_vals, mode='lines', name='Simulated MSD'))
        coeffs = np.polyfit(t_vals, msd_vals, 1)
        line_fit = coeffs[0]*t_vals + coeffs[1]
        fig_msd.add_trace(go.Scatter(x=t_vals, y=line_fit, mode='lines', line=dict(dash='dash'), name=f'Linear fit\nD={D_num:.6f}'))
        fig_msd.update_layout(title="Mean Squared Displacement (MSD)", xaxis_title="t (steps)", yaxis_title="MSD(t)")
        st.plotly_chart(fig_msd, use_container_width=True)

if __name__ == "__main__":
    main()
