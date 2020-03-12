import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {Memory} from "../../models";

@Component({
    selector: 'memory-add-dialog',
    templateUrl: 'memory-add-dialog.html',
})
export class MemoryAddDialogComponent {

    constructor(
        public dialogRef: MatDialogRef<MemoryAddDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: Memory) {

    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}