pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Window
import QtQuick.Templates as T

/*!
    GlassComboBox - T.ComboBox (docs/GLASS_DESIGN.md §2.3).  `model` is a list
    of `{value, text}` objects (textRole/valueRole "text"/"value"; a plain
    string list works with both roles set to "").  Bind `value` to the bridge
    property; the current index follows it and `selected(value)` fires when the
    user picks an item (the component never assigns `value` itself, so the
    binding survives).  Closed state = "fill" material; the popup is a
    T.Popup in the overlay (modal: false, dim: false) whose direction and height
    are computed once on open so the list scrolls instead of being clipped.
    Keyboard: Up/Down, Enter, Esc, type-ahead (built into T.ComboBox).
    `fontPreview` renders each item (and the closed state) in the font it
    names - the GlassFontComboBox variant.
*/
T.ComboBox {
    id: control

    property var value: undefined
    property string label: ""
    property string hint: ""
    property bool fontPreview: false
    signal selected(var value)

    readonly property real rowHeight: Math.max(Theme.controlH - 4, fm.height + 8)
    readonly property color inkColor: (Theme.mode === "hc" && !enabled) ? palette.disabled.windowText : Theme.ink
    // near-opaque sheet of the current material polarity: a translucent fill over
    // page text would not be legible
    readonly property color popupFill: Theme.mode === "hc" ? Theme.sysPal.base
        : Qt.rgba(0.16 + 0.81 * Theme.inkValue, 0.17 + 0.81 * Theme.inkValue, 0.20 + 0.80 * Theme.inkValue, 0.94)
    readonly property color highlightFill: Theme.mode === "hc" ? Theme.sysPal.highlight
        : Qt.rgba(Theme.ink.r, Theme.ink.g, Theme.ink.b, 0.10)

    textRole: "text"
    valueRole: "value"

    implicitWidth: Math.max(implicitBackgroundWidth + leftInset + rightInset,
                            implicitContentWidth + leftPadding + rightPadding, 160)
    implicitHeight: Math.max(Theme.controlH, fm.height + 2 * Theme.padY,
                             implicitContentHeight + topPadding + bottomPadding)
    leftPadding: 12
    rightPadding: 12 + (indicator ? indicator.width + Theme.gap : 0)
    topPadding: Theme.padY
    bottomPadding: Theme.padY
    font.family: Theme.family
    font.pointSize: Theme.fontBody
    focusPolicy: Qt.StrongFocus
    hoverEnabled: true
    opacity: enabled || Theme.mode === "hc" ? 1 : 0.45

    Accessible.role: Accessible.ComboBox
    Accessible.name: label
    Accessible.description: hint

    FontMetrics { id: fm; font: control.font }

    // ---- model helpers (independent of the delegate model internals)
    function textOf(d) {
        if (d === null || d === undefined)
            return ""
        if (typeof d === "object" && control.textRole.length > 0)
            return String(d[control.textRole])
        return String(d)
    }
    function valueOf(d) {
        if (d === null || d === undefined)
            return d
        if (typeof d === "object" && control.valueRole.length > 0)
            return d[control.valueRole]
        return d
    }
    function indexFor(v) {
        var m = control.model
        if (!m || m.length === undefined)
            return -1
        for (var i = 0; i < m.length; ++i)
            if (valueOf(m[i]) === v)
                return i
        return -1
    }
    function sync() {
        var i = indexFor(value)
        if (i >= 0 && i !== currentIndex)
            currentIndex = i
    }
    onValueChanged: sync()
    onModelChanged: sync()
    Component.onCompleted: sync()
    onActivated: function (index) {
        var m = control.model
        if (m && m.length !== undefined && index >= 0 && index < m.length)
            selected(valueOf(m[index]))
    }

    // ---- press compress .97 (combos and toggles)
    onDownChanged: {
        pressIn.stop()
        pressOut.stop()
        if (Theme.reduceMotion) {
            scale = 1
        } else if (down) {
            pressIn.start()
        } else {
            pressOut.start()
        }
    }
    NumberAnimation { id: pressIn; target: control; property: "scale"; to: 0.97; duration: Theme.pressInMs; easing.type: Easing.OutCubic }
    SpringAnimation { id: pressOut; target: control; property: "scale"; to: 1; spring: Theme.springPress.spring; damping: Theme.springPress.damping }

    // ---- one-shot popup placement (§2.3): direction + height, never a binding.
    // Bounded by the slab rect (Theme.slabRect), not the window: the transparent
    // shadow margin must never carry a popup sheet.
    function placePopup() {
        var top = control.mapToItem(null, 0, 0).y
        var slabBottom = Theme.slabRect.y + Theme.slabRect.height
        var spaceBelow = slabBottom - (top + control.height) - Theme.pad
        var spaceAbove = top - Theme.slabRect.y - Theme.pad
        var want = control.count * control.rowHeight + 2 * popup.padding
        var h = Math.max(control.rowHeight + 2 * popup.padding, Math.min(want, Math.max(spaceBelow, spaceAbove)))
        popup.height = h
        popup.y = want <= spaceBelow ? control.height + 2 : -(h + 2)
    }

    background: Item {
        implicitWidth: 160
        implicitHeight: Theme.controlH
        GlassShadow {
            anchors.fill: parent
            radius: Theme.radiusControl
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.16)
        }
        GlassSurface {
            anchors.fill: parent
            variant: "fill"
            radius: Theme.radiusControl
            fillColor: Theme.mode !== "hc" && control.hovered && control.enabled
                       ? Qt.rgba(1, 1, 1, Math.min(1, Theme.fillControl.a + 0.06)) : Theme.fillControl
            borderColor: control.enabled ? Theme.borderColor : control.palette.disabled.windowText
            Behavior on fillColor { ColorAnimation { duration: Theme.motionDuration } }
        }
        FocusRing { ringRadius: Theme.radiusControl; active: control.visualFocus }
    }

    contentItem: Text {
        text: control.displayText
        font.family: control.fontPreview && control.displayText.length > 0 ? control.displayText : control.font.family
        font.pointSize: control.font.pointSize
        color: control.inkColor
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Glyph {
        x: control.width - width - 12
        y: (control.height - height) / 2
        name: "chevron-down"
        color: control.inkColor
        rotation: control.popup.visible ? 180 : 0
        Behavior on rotation { NumberAnimation { duration: Theme.fade; easing.type: Easing.OutCubic } }
    }

    delegate: T.ItemDelegate {
        id: item
        required property int index
        required property var modelData
        width: ListView.view ? ListView.view.width : control.width
        height: control.rowHeight
        text: control.textOf(modelData)
        highlighted: control.highlightedIndex === index
        hoverEnabled: true
        font: control.font
        leftPadding: 10
        rightPadding: 10
        Accessible.role: Accessible.ListItem
        Accessible.name: text
        background: Rectangle {
            radius: Theme.concentric(Theme.radiusCard, 4)
            color: item.highlighted ? control.highlightFill : "transparent"
        }
        contentItem: Item {
            Text {
                anchors.left: parent.left
                anchors.right: check.left
                anchors.rightMargin: 6
                anchors.verticalCenter: parent.verticalCenter
                text: item.text
                font.family: control.fontPreview ? item.text : control.font.family
                font.pointSize: control.font.pointSize
                color: Theme.mode === "hc" && item.highlighted ? Theme.sysPal.highlightedText : Theme.ink
                elide: Text.ElideRight
            }
            Glyph {
                id: check
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                name: "check"
                visible: item.index === control.currentIndex
                color: Theme.mode === "hc" && item.highlighted ? Theme.sysPal.highlightedText : Theme.ink
            }
        }
    }

    popup: T.Popup {
        id: popup
        popupType: T.Popup.Item
        modal: false
        dim: false
        closePolicy: T.Popup.CloseOnEscape | T.Popup.CloseOnPressOutsideParent
        padding: 4
        width: control.width
        onAboutToShow: control.placePopup()
        onOpened: list.positionViewAtIndex(control.currentIndex, ListView.Contain)

        contentItem: ListView {
            id: list
            clip: true
            reuseItems: true
            implicitHeight: contentHeight
            model: popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex
            highlightFollowsCurrentItem: true
            highlightMoveDuration: 0
            highlightResizeDuration: 0
            boundsBehavior: Flickable.StopAtBounds
            T.ScrollBar.vertical: GlassScrollBar {}
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
                fillColor: control.popupFill
            }
        }
    }
}
