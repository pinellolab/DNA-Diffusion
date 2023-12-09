{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    # nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-trusted-public-keys = [
      "dnadiffusion.cachix.org-1:P20JWJrVBiN5iPBnzJ4UiqLVghGBCYOicXxltPRLaEY="
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
    ];
    extra-substituters = [
      "https://dnadiffusion.cachix.org"
      "https://devenv.cachix.org"
    ];
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    systems,
    ...
  } @ inputs: let
    forEachSystem = nixpkgs.lib.genAttrs (import systems);
  in {
    packages = forEachSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devenv-up = self.devShells.${system}.default.config.procfileScript;
    });

    devShells =
      forEachSystem
      (system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            {
              packages = with pkgs; [
                atuin
                bat
                bedtools
                gcc
                gh
                git
                gnumake
                htslib
                lazygit
                poethepoet
                poetry
                python310
                ripgrep
                starship
                tree
                zlib
                zlib.dev
                zsh
              ];

              dotenv = {
                enable = true;
                filename = ".env";
                # disableHint = true;
              };

              pre-commit.hooks = {
                alejandra.enable = true;
                ruff.enable = true;
                # pyright.enable = true;
              };

              difftastic.enable = true;

              # languages.python = {
              #   enable = true;
              #   package = pkgs.python310;
              #   poetry = {
              #     enable = true;
              #     activate.enable = true;
              #     install = {
              #       enable = true;
              #       installRootPackage = true;
              #       groups = [
              #         "lint"
              #         "test"
              #         "docs"
              #         "workflows"
              #       ];
              #     };
              #   };
              # };
            }
          ];
        };
      });
  };
}
