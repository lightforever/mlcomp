import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ProjectAddData} from "../models";

@Component({
    selector: 'project-add-dialog',
    templateUrl: 'project-add-dialog.html',
})
export class ProjectAddDialogComponent {

    constructor(
        public dialogRef: MatDialogRef<ProjectAddDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ProjectAddData) {
    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}