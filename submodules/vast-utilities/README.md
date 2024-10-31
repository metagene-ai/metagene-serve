# vast-utilities
Repo of utility files for training, inference, testing on [vast instances](https://vast.ai/).

Upon creating a new vast instance, run the following shell commands to install [Github CLI](https://cli.github.com/).
Remember to login via web one-time password, Github token may result in error.
```shell
sudo apt install gh && gh auth login
```

Then, go change permission for execution, which basically chmod +x for all .sh files within the project folder
```shell
find path/to/project/ -type f -name "*.sh" -exec chmod +x {} +
```
