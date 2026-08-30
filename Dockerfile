# ==============================================================================
# DOCKERFILE: MT5 AI REINFORCEMENT LEARNING TRADING AGENT (DEBIAN 12 HEADLESS)
# ==============================================================================
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV WINEPREFIX=/root/.wine
ENV WINEARCH=win64
ENV WINEDEBUG=-all
ENV DISPLAY=:99

WORKDIR /app

# 1. Install Dependencies & WINE
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl ca-certificates gnupg2 software-properties-common \
        xvfb xauth cabextract winbind procps unzip && \
    mkdir -pm755 /etc/apt/keyrings && \
    wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key && \
    wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/debian/dists/bookworm/winehq-bookworm.sources && \
    apt-get update && \
    (apt-get install -y --install-recommends winehq-stable || apt-get install -y wine wine64 wine32) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Setup Virtual Framebuffer & Wine environment
RUN (Xvfb :99 -screen 0 1024x768x16 &) && \
    wineboot -i && \
    sleep 3

# 3. Install Windows Python 3.10 inside Wine
RUN wget -q "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe" -O /tmp/python-installer.exe && \
    (Xvfb :99 -screen 0 1024x768x16 &) && \
    wine /tmp/python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 && \
    sleep 5 && \
    rm -f /tmp/python-installer.exe

# 4. Install MT5 Terminal inside Wine
RUN wget -q "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" -O /tmp/mt5setup.exe && \
    (Xvfb :99 -screen 0 1024x768x16 &) && \
    wine /tmp/mt5setup.exe /auto || true && \
    sleep 8 && \
    rm -f /tmp/mt5setup.exe

# 5. Copy dependencies and install python packages in Wine
COPY requirements.txt /app/
RUN (Xvfb :99 -screen 0 1024x768x16 &) && \
    wine python -m pip install --upgrade pip setuptools wheel && \
    wine python -m pip install -r /app/requirements.txt

# 6. Copy application files
COPY . /app/

# 7. Entrypoint
CMD ["/bin/bash", "-c", "Xvfb :99 -screen 0 1024x768x16 -nolisten tcp & sleep 2 && DISPLAY=:99 wine python mt5_live_trader.py --mode DEMO"]
