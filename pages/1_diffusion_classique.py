import numpy as np
import streamlit as st
import plotly.graph_objs as go

def kmc_diffusion_2D(num_steps, Gamma1, Gamma2, a=1.0):
    d1 = a
    d2 = np.sqrt(2) * a
    Gamma_tot = Gamma1 + Gamma2
    directions_type1 = np.array([[d1, 0], [-d1, 0], [0, d1], [0, -d1]])
    directions_type2 = np.array([[d2, d2], [d2, -d2], [-d2, d2], [-d2, -d2]])
    positions = np.zeros((num_steps+1, 2))
    prob_gamma1 = Gamma1 / Gamma_tot
    for i in range(1, num_steps+1):
        r = np.random.rand()
        if r < prob_gamma1:
            step = directions_type1[np.random.choice(4)]
        else:
            step = directions_type2[np.random.choice(4)]
        positions[i] = positions[i-1] + step
    return positions

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
    D_est = coeffs[0] / 4
    return D_est

def analytical_diffusion_coefficient(Gamma1, Gamma2, a=1.0):
    d1 = a
    d2 = np.sqrt(2) * a
    D_analyt = 0.25 * (Gamma1 * d1**2 + Gamma2 * d2**2)
    return D_analyt

def main():
    st.title("Classical Monte Carlo Diffusion 2D")
    st.set_page_config(layout="wide")
    num_steps = st.slider("Number of simulation steps", 1000, 50000, 10000, step=1000)
    col_ratio1, col_ratio2 = st.columns(2)
    with col_ratio1:
        ratio1 = st.slider("Gamma1/Gamma ratio (case 1)", 0.0, 1.0, 0.25, step=0.01)
    with col_ratio2:
        ratio2 = st.slider("Gamma1/Gamma ratio (case 2)", 0.0, 1.0, 0.75, step=0.01)

    if st.button("Run simulation"):
        results = []
        for ratio in [ratio1, ratio2]:
            Gamma1 = ratio
            Gamma2 = 1 - ratio
            traj = kmc_diffusion_2D(num_steps, Gamma1, Gamma2)
            t_vals, msd_vals = compute_msd(traj)
            D_num = estimate_diffusion_coefficient(t_vals, msd_vals)
            D_ana = analytical_diffusion_coefficient(Gamma1, Gamma2)
            results.append((ratio, traj, t_vals, msd_vals, D_num, D_ana))
        st.session_state['classical_results'] = results

    # Display persistent results if available
    if 'classical_results' in st.session_state:
        results = st.session_state['classical_results']
        with st.container():
            st.markdown("<h3 style='text-align:center;'>Simulation Results</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            for idx, (ratio, traj, t_vals, msd_vals, D_num, D_ana) in enumerate(results):
                with [col1, col2][idx]:
                    st.markdown(f"<h4 style='text-align:center;'>Gamma1/Gamma = {ratio:.2f} &nbsp;&nbsp; Gamma2/Gamma = {1-ratio:.2f}</h4>", unsafe_allow_html=True)
                    st.metric("Estimated numerical diffusion D", f"{D_num:.6f}")
                    st.metric("Analytical diffusion D", f"{D_ana:.6f}")
                    st.markdown("<hr>", unsafe_allow_html=True)
                    x_vals, y_vals = traj[:,0], traj[:,1]
                    fig_traj = go.Figure()
                    fig_traj.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', line=dict(color='lightgray'), marker=dict(size=3), name='Trajectory'))
                    fig_traj.add_trace(go.Scatter(x=[x_vals[0]], y=[y_vals[0]], mode='markers', marker=dict(color='green', size=10), name='Start'))
                    fig_traj.add_trace(go.Scatter(x=[x_vals[-1]], y=[y_vals[-1]], mode='markers', marker=dict(color='red', size=10), name='End'))
                    fig_traj.update_layout(title=f"2D Diffusion Trajectory (Gamma1/Gamma={ratio:.2f})", xaxis_title="x", yaxis_title="y", yaxis_scaleanchor='x', margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_traj, use_container_width=True)
                    fig_msd = go.Figure()
                    fig_msd.add_trace(go.Scatter(x=t_vals, y=msd_vals, mode='lines', name='Simulated MSD'))
                    coeffs = np.polyfit(t_vals, msd_vals, 1)
                    line_fit = coeffs[0]*t_vals + coeffs[1]
                    fig_msd.add_trace(go.Scatter(x=t_vals, y=line_fit, mode='lines', line=dict(dash='dash'), name=f'Linear fit<br>D={D_num:.6f}'))
                    fig_msd.update_layout(title="Mean Squared Displacement MSD(t)", xaxis_title="t (steps)", yaxis_title="MSD(t)", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_msd, use_container_width=True)

if __name__ == "__main__":
    main()
