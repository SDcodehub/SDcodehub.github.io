---
layout: post
title: Setting Up WSL Terminal with Zsh and Oh-My-Zsh on Windows
date: 2023-10-23
author: Sagar Desai
categories: [dev]
tags: [linux, vs code, zsh, bash]
image: /path/to/featured-image.jpg
---
### Setting Up WSL Terminal with Zsh and Oh-My-Zsh on Windows

#### Step 1: Open the Linux Terminal
- Open the Windows Subsystem for Linux (WSL) terminal by searching for "WSL" in the Windows Start menu.

#### Step 2: Install Oh-My-Zsh
- Visit the [Oh-My-Zsh website](https://ohmyz.sh/).
- Run the following command in your WSL terminal to install Oh-My-Zsh:

  ```shell
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
  ```

  This command installs Oh-My-Zsh, a framework for managing Zsh configurations and plugins.

#### Step 3: Install Zsh (if necessary)
- If you encounter an error during Oh-My-Zsh installation, you may need to install Zsh. Run the following command:

  ```shell
  sudo apt-get install zsh
  ```

  - Zsh is a highly customizable shell, similar to Bash, that Oh-My-Zsh is built on.

#### Step 4: Open WSL in Visual Studio Code
- Launch Visual Studio Code.
- Use the tilde (~) key and Enter to navigate to your home directory.

#### Step 5: List Hidden Files
- To view hidden files in your home directory, run:

  ```shell
  ls -a
  ```

#### Step 6: Edit .zshrc
- The `.zshrc` file is the configuration file for Zsh, similar to `.bashrc` for Bash. You can edit it with a text editor like Nano or Visual Studio Code.
- Open the `.zshrc` file:

  ```shell
  code ~/.zshrc
  ```

- Modify the `ZSH_THEME` variable to set your preferred theme. For example:

  ```shell
  ZSH_THEME="bira"
  ```

  This changes the appearance of your WSL terminal.

#### Step 7: Enable Plugins
- Enhance your terminal by enabling various plugins in the `.zshrc` file. For example:

  ```shell
  # Add wisely, as too many plugins can slow down shell startup.
  plugins=(git web-search python pyenv virtualenv pip)
  source $ZSH/oh-my-zsh.sh
  ```

  - After making changes to `.zshrc`, save the file and exit the editor.

#### Step 8: Install Zsh-Autosuggestions (Auto-Completion)
- Install the Zsh-Autosuggestions plugin for auto-completion. Refer to the [GitHub repository](https://github.com/zsh-users/zsh-autosuggestions) for installation instructions.
- Install the plugin using these commands:

  ```shell
  git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
  ```

  Add `zsh-autosuggestions` to the list of activated plugins in your `.zshrc`:

  ```shell
  plugins=(git web-search python pyenv virtualenv pip zsh-autosuggestions)
  ```

  Save your changes and exit the editor.

#### Step 9: Install Zsh-Syntax-Highlighting
- Install the Zsh-Syntax-Highlighting plugin for syntax highlighting. Refer to the [GitHub repository](https://github.com/zsh-users/zsh-syntax-highlighting/blob/master/INSTALL.md) for installation instructions.
- Install the plugin using these commands:

  ```shell
  git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
  ```

  Add `zsh-syntax-highlighting` to the list of activated plugins in your `.zshrc`:

  ```shell
  plugins=(git web-search python pyenv virtualenv pip zsh-autosuggestions zsh-syntax-highlighting)
  ```

  Save your changes and exit the editor.

#### Step 10: Restart the Zsh Terminal
- To apply the changes you've made, you need to restart the Zsh terminal. Simply type:

  ```shell
  zsh
  ```

  This will reload the Zsh configuration and apply the new theme and plugins.

Your WSL terminal is now set up with Zsh and Oh-My-Zsh, making it more user-friendly and feature-rich for Python development. Enjoy your enhanced command-line experience!