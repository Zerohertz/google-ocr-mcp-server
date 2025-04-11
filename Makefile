SHELL := /bin/bash

.PHONY: deploy
deploy:
ifndef tag
	$(error tag is not set)
endif
ifneq ($(shell echo $(tag) | grep -E '^v'),$(tag))
	$(error tag must start with 'v')
endif
	echo $(tag)
	sed -i '' 's|^\(version = \).*|\1"$(tag)"|g' pyproject.toml
	uv sync
	git add pyproject.toml
	git add uv.lock
	git commit -m ":ship: release: $(tag)"
	git tag -a $(tag) -m ":ship: release: $(tag)"
	git push origin main
	git push origin $(tag)

