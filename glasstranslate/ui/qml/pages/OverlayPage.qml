pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T
import ".."

/*!
    OverlayPage (docs/GLASS_DESIGN.md §2.3): card "Glass" (background opacity
    slider previewing live, font, hide original) and card "Typesetting" (manga
    mode, uppercase - disabled while manga mode is off); hints are the old
    tooltips.
*/
// qmllint disable unqualified
Flickable {
    id: page

    contentWidth: width
    contentHeight: column.implicitHeight
    clip: true
    boundsBehavior: Flickable.StopAtBounds
    flickableDirection: Flickable.VerticalFlick
    T.ScrollBar.vertical: GlassScrollBar {}

    Accessible.role: Accessible.Pane
    Accessible.name: "Overlay"

    Column {
        id: column
        width: page.width
        spacing: Theme.gap

        GlassCard {
            title: "Glass"
            FormRow {
                label: "Background opacity"
                GlassSlider {
                    id: opacitySlider
                    objectName: "opacitySlider"
                    from: 0
                    to: 100
                    stepSize: 1
                    live: true
                    valueText: Math.round(value) + " %"
                    Binding on value { value: Math.round(bridge.overlayOpacity * 100) }
                    onMoved: bridge.overlayOpacity = value / 100
                }
            }
            FormRow {
                label: "Font"
                GlassFontComboBox {
                    objectName: "fontCombo"
                    model: bridge.fontFamilies
                    value: bridge.fontFamily
                    onSelected: function (v) { bridge.fontFamily = v }
                }
            }
            FormRow {
                label: "Hide original"
                hint: "Paint a background-coloured box under the translation so the source text disappears."
                GlassToggle {
                    id: hideOriginal
                    objectName: "hideOriginalToggle"
                    text: "Cover the original text"
                    Binding on checked { value: bridge.hideOriginal }
                    onToggled: bridge.hideOriginal = hideOriginal.checked
                }
            }
        }

        GlassCard {
            title: "Typesetting"
            FormRow {
                label: "Manga mode"
                hint: "Group the columns of a speech bubble into one block, translate the whole utterance, erase the original and letter the translation in the bundled Anime Ace font, flowed to the bubble outline."
                GlassToggle {
                    id: mangaMode
                    objectName: "mangaModeToggle"
                    text: "Group bubbles, comic lettering"
                    Binding on checked { value: bridge.mangaMode }
                    onToggled: bridge.mangaMode = mangaMode.checked
                }
            }
            FormRow {
                label: "Uppercase"
                hint: "Letter manga-mode blocks in capitals, as printed English comics do."
                GlassToggle {
                    id: uppercase
                    objectName: "uppercaseToggle"
                    text: "Capitals"
                    enabled: bridge.mangaMode
                    Binding on checked { value: bridge.uppercase }
                    onToggled: bridge.uppercase = uppercase.checked
                }
            }
        }
    }
}
