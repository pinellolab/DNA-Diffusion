FROM ghcr.io/cachix/devenv:latest

RUN echo 'extra-substituters = https://devenv.cachix.org' >> /etc/nix/nix.conf && \
    echo 'extra-trusted-public-keys = devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=' >> /etc/nix/nix.conf
