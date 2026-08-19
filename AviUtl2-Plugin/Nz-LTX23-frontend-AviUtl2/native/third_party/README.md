# Vendored third-party dependencies

These are checked into the repository so the native plugin builds without any
network access or NuGet restore step. Only the minimal set of files needed to
build/link is kept.

| Component | Version | Source | Files kept |
|---|---|---|---|
| Microsoft.Web.WebView2 | **1.0.4022.49** | https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2 | `webview2/include/WebView2.h`, `webview2/include/WebView2EnvironmentOptions.h`, `webview2/x64/WebView2LoaderStatic.lib` |
| Microsoft.Windows.ImplementationLibrary (WIL) | **1.0.260126.7** | https://www.nuget.org/api/v2/package/Microsoft.Windows.ImplementationLibrary | `wil/include/wil/*` |
| nlohmann/json | **3.12.0** | https://github.com/nlohmann/json/releases/latest/download/json.hpp | `nlohmann/json.hpp` |
| doctest | (see doctest/doctest.h header) | https://github.com/doctest/doctest | `doctest/doctest.h` |

## Notes

- `WebView2LoaderStatic.lib` is a static loader; linking it means no separate
  `WebView2Loader.dll` needs to be shipped. The actual WebView2 runtime is the
  Evergreen runtime installed system-wide (part of modern Windows / Edge).
- `WebView2.h` references `EventToken.h`, which is provided by the Windows SDK,
  not vendored here.
- These `.lib` files are force-tracked in `.gitignore` (see the negation rules
  for `native/third_party/**`).
- Licenses: WebView2 SDK (proprietary, redistributable per NuGet EULA), WIL
  (MIT), nlohmann/json (MIT), doctest (MIT).
