import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {SyncStart} from "../models";

@Component({
    selector: 'sync-dialog',
    templateUrl: 'sync-dialog.html',
})
export class SyncDialogComponent {

    constructor(
        public dialogRef: MatDialogRef<SyncDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: SyncStart) {
        if(data.projects && data.projects.length >0){
            data.project = data.projects[0];
        }
    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}