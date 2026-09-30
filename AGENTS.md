# AGENTS.md

## Development environment setup
- Run `make config BUILD=release` to configure the project. (Default build target is debug)
- Run `make -j$(nproc)` to compile, No need to run `make clean` after making changes.

## Testing
- Use `./build/kappai-test reference_backend target_backend` to test for any regressions. (e.g. cpu_scalar vulkan)
- Avoid running `./build/kappai-test --bench backend`, All performance critical testing should be done by the user to avoid OS background noise.

## Coding rules
- Avoid writing comments, Good code self documents.
- Environment variables should only be used during testing and should not reach production.
- Avoid duplication, If something that does 90% of what you need exists try expanding it's functionality over re-implementing it + strapping on a 3 line change.
- Avoid single line wrappers/functions.
- If a macro is longer than 2-3 lines it's best to turn it into a function.

## PR instructions
- Title format: Short description of what the commit does.
- Description format: Short description of what the commit contains/addresses.
- Run `make format` followed by `make -j$(nproc)` then test before making a PR.
- Never run any git command yourself without human confirmation.
- Before opening a PR do a review of the changes and assess if there's a better/cleaner way to do what the PR aims to achieve. (e.g. Is this the most optimal approach?)
