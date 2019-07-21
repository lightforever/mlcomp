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
        if(!data.class_names)
        {
            data.class_names = 'default: [\n\n]';
        }
        if(!data.ignore_folders){
            data.ignore_folders = '[\n\n]'
        }
    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}