import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {LayoutAddData} from "../../models";

@Component({
    selector: 'layout-add-dialog',
    templateUrl: 'layout-add-dialog.html',
})
export class LayoutAddDialogComponent {

    constructor(
        public dialogRef: MatDialogRef<LayoutAddDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: LayoutAddData) {

    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}