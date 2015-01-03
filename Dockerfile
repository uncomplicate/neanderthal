FROM ruby:2.0

ADD . /usr/src/app
WORKDIR /usr/src/app

RUN bundle install