#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROJECT_VERSION="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
VERSION="${LEAPS_VERSION:-$PROJECT_VERSION}"
VERSION="${VERSION#v}"
if [[ ! "$VERSION" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]; then
  echo "LEAPS_VERSION must contain two or three numeric components (received: $VERSION)" >&2
  exit 1
fi
export LEAPS_VERSION="$VERSION"

python -m pip install --upgrade 'PyInstaller>=6.14,<7'
python -m PyInstaller --noconfirm --clean packaging/LEAPS-macos.spec

APP="dist/LEAPS.app"
if [[ ! -d "$APP" ]]; then
  echo "Expected $APP was not produced" >&2
  exit 1
fi

INFO_PLIST="$APP/Contents/Info.plist"
set_plist_string() {
  local key="$1"
  local value="$2"
  /usr/libexec/PlistBuddy -c "Set :$key $value" "$INFO_PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Add :$key string $value" "$INFO_PLIST"
}

# Keep the privacy identity stable across releases and explain the native
# macOS folder-access prompts shown for protected locations.
set_plist_string CFBundleDisplayName "LEAPS"
set_plist_string CFBundleName "LEAPS"
set_plist_string CFBundleIdentifier "org.leaps.exoplanet"
set_plist_string CFBundleShortVersionString "$VERSION"
set_plist_string CFBundleVersion "$VERSION"
set_plist_string NSDocumentsFolderUsageDescription \
  "LEAPS needs access to your observing-run folder to read FITS images and save project results beside them."
set_plist_string NSDesktopFolderUsageDescription \
  "LEAPS needs access when an observing run is stored on your Desktop."
set_plist_string NSDownloadsFolderUsageDescription \
  "LEAPS needs access when an observing run is stored in Downloads."
set_plist_string NSNetworkVolumesUsageDescription \
  "LEAPS needs access when an observing run is stored on a network volume."
set_plist_string NSRemovableVolumesUsageDescription \
  "LEAPS needs access when an observing run is stored on a removable drive."

if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  while IFS= read -r -d '' binary; do
    codesign --force --options runtime --timestamp \
      --sign "$APPLE_SIGNING_IDENTITY" "$binary"
  done < <(find "$APP/Contents" -type f \( -name '*.dylib' -o -name '*.so' \) -print0)
  while IFS= read -r framework; do
    codesign --force --options runtime --timestamp \
      --sign "$APPLE_SIGNING_IDENTITY" "$framework"
  done < <(find "$APP/Contents" -depth -type d -name '*.framework')
  codesign --force --options runtime --timestamp \
    --entitlements packaging/entitlements.plist \
    --sign "$APPLE_SIGNING_IDENTITY" "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
else
  # Editing Info.plist invalidates PyInstaller's initial ad-hoc signature. Keep
  # local and unsigned CI artifacts launchable after applying metadata.
  codesign --force --deep --sign - "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
fi

APP_EXECUTABLE="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$INFO_PLIST")"
"$APP/Contents/MacOS/$APP_EXECUTABLE" --packaging-self-test

mkdir -p artifacts
hdiutil create -volname LEAPS -srcfolder "$APP" -ov -format UDZO artifacts/LEAPS-Apple-Silicon.dmg

if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  codesign --force --timestamp --sign "$APPLE_SIGNING_IDENTITY" artifacts/LEAPS-Apple-Silicon.dmg
fi

if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
  xcrun notarytool submit artifacts/LEAPS-Apple-Silicon.dmg --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
  xcrun stapler staple artifacts/LEAPS-Apple-Silicon.dmg
  spctl --assess --type open --context context:primary-signature --verbose artifacts/LEAPS-Apple-Silicon.dmg
fi
