pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T

/*!
    WindowMenu - the Alt+Space replacement for the native system menu of a
    frameless window (docs/GLASS_DESIGN.md §1.1): Minimise, Move, Close (no
    "Size" - the window is fixed-size).  A T.Menu (keyboard: Up/Down, Enter,
    Esc) on the same near-opaque sheet as the combo popup.  Main.qml wires the
    signals to the bridge.
*/
T.Menu {
    id: menu

    signal minimizeRequested()
    signal moveRequested()
    signal closeRequested()

    readonly property color sheetFill: Theme.mode === "hc" ? Theme.sysPal.base
        : Qt.rgba(0.16 + 0.81 * Theme.inkValue, 0.17 + 0.81 * Theme.inkValue, 0.20 + 0.80 * Theme.inkValue, 0.94)
    readonly property color highlightFill: Theme.mode === "hc" ? Theme.sysPal.highlight
        : Qt.rgba(Theme.ink.r, Theme.ink.g, Theme.ink.b, 0.10)
    readonly property real rowHeight: Math.max(Theme.controlH - 4, fm.height + 8)

    popupType: T.Popup.Item
    modal: false
    dim: false
    focus: true
    closePolicy: T.Popup.CloseOnEscape | T.Popup.CloseOnPressOutside
    padding: 4
    implicitWidth: Math.max(180, Math.round(160 * Theme.textScale))
    implicitHeight: contentItem.implicitHeight + topPadding + bottomPadding
    font.family: Theme.family
    font.pointSize: Theme.fontBody
    // opened by keyboard only (Alt+Space): Windows highlights the first item of a
    // keyboard-opened menu so Enter / Down work immediately
    onOpened: currentIndex = 0

    FontMetrics { id: fm; font: menu.font }

    contentItem: ListView {
        implicitHeight: contentHeight
        model: menu.contentModel
        currentIndex: menu.currentIndex
        interactive: false
        clip: true
    }

    background: Item {
        GlassShadow {
            anchors.fill: parent
            radius: Theme.radiusCard
            sigma: 10
            offsetY: 6
            color: Qt.rgba(0, 0, 0, 0.28)
        }
        GlassSurface {
            anchors.fill: parent
            variant: "fill"
            radius: Theme.radiusCard
            fillColor: menu.sheetFill
        }
    }

    component Entry: T.MenuItem {
        id: entry
        width: ListView.view ? ListView.view.width : menu.availableWidth
        height: menu.rowHeight
        leftPadding: 10
        rightPadding: 10
        font: menu.font
        hoverEnabled: true
        Accessible.role: Accessible.MenuItem
        Accessible.name: text
        background: Rectangle {
            radius: Theme.concentric(Theme.radiusCard, 4)
            color: entry.highlighted || entry.hovered ? menu.highlightFill : "transparent"
        }
        contentItem: Text {
            text: entry.text
            font: entry.font
            color: Theme.mode === "hc" && (entry.highlighted || entry.hovered) ? Theme.sysPal.highlightedText : Theme.ink
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
    }

    Entry { text: "Minimise"; onTriggered: menu.minimizeRequested() }
    Entry { text: "Move"; onTriggered: menu.moveRequested() }
    Entry { text: "Close"; onTriggered: menu.closeRequested() }
}
