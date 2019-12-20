FROM node:slim

WORKDIR /app

RUN apt-get update \
  && apt-get upgrade -y \
  && curl -o- -L https://yarnpkg.com/install.sh | sh

ENV PATH=$HOME/.yarn/bin:$HOME/.config/yarn/global/node_modules/.bin:$PATH

CMD ["yarn", "start"]