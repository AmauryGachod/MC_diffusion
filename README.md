# Monte Carlo Diffusion Dashboard

This Streamlit dashboard allows you to simulate and visualize classical and modified Monte Carlo diffusion processes in 2D.

## Features
- Classical diffusion (standard KMC simulation)
- Modified diffusion (custom parameters, question II)
- Interactive controls and persistent plots
- Beautiful, responsive UI

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AmauryGachod/MC_diffusion.git
   cd MC_diffusion/streamlit_dashboard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard
```bash
streamlit run main.py
```

- The dashboard will open in your browser.
- Use the sidebar to navigate between pages.





## Folder Structure
```
streamlit_dashboard/
├── main.py
├── requirements.txt
├── README.md
├── pages/
│   ├── 1_diffusion_classique.py
│   └── 2_diffusion_modifiee.py
├── pages.toml
└── .streamlit/
    └── config.toml
```

## License
Specify your license here (MIT, GPL, etc.)

---
Feel free to open issues or pull requests for improvements!
