# marigold-examples

Example applications for [Marigold](https://github.com/bayinfosys/marigold),
published as a registry at [examples.marigold.run](https://examples.marigold.run/).
Each directory holding a `marigold.toml` is one application.

```sh
marigold registry add https://examples.marigold.run/
```

What each application is for, and where to start, is in the
[Examples](https://marigold.run/docs/examples.html) documentation. Each
application's catalogue page carries its README, its models and the
install commands.

## Publishing

`.github/workflows/publish.yaml` builds the registry on every push and
publishes it on a `v*` tag. See
[Publishing](https://marigold.run/docs/publishing.html).
