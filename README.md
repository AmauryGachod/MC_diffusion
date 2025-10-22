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
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name/streamlit_dashboard
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

## Deployment (Optional)
You can deploy your dashboard for free using [Streamlit Cloud](https://streamlit.io/cloud):
1. Push your code to GitHub (see below).
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your repo.
3. Set the main file to `main.py`.
4. Share the app link with others!

## How to Upload Your Code to GitHub
1. [Create a GitHub account](https://github.com/join) if you don't have one.
2. [Create a new repository](https://github.com/new) (public or private).
3. In your terminal, initialize git in your project folder:
   ```bash
   cd /path/to/streamlit_dashboard
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/your-username/your-repo-name.git
   git push -u origin main
   ```
4. Your code is now on GitHub!

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
