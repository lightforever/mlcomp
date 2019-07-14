import {Component, Inject} from '@angular/core';
import {Paginator} from "../paginator";
import {Project, ProjectAddData, ProjectFilter} from "../models";
import {Location} from '@angular/common';
import {
    MAT_DIALOG_DATA,
    MatDialog,
    MatDialogRef,
    MatIconRegistry
} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";
import {DialogComponent} from "../dialog/dialog.component";
import {Helpers} from "../helpers";
import {ProjectService} from "./project.service";

@Component({
    selector: 'app-project',
    templateUrl: './project.component.html',
    styleUrls: ['./project.component.css']
})
export class ProjectComponent extends Paginator<Project> {

    displayed_columns: string[] = ['name', 'dag_count', 'last_activity', 'img_size', 'file_size', 'links'];
    name: string;

    constructor(protected service: ProjectService, protected location: Location,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
                public dialog: MatDialog
    ) {
        super(service, location);
        iconRegistry.addSvgIcon('remove',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/trash.svg'));
    }

    get_filter() {
        let res = new ProjectFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        return res;
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }


    remove(element: Project) {
        const dialogRef = this.dialog.open(DialogComponent, {
            width: '550px', height: '200px',
            data: {'message': 'The all content will be deleted. Do you want to continue?'}
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.remove(element.id).subscribe(data => {
                    this.change.emit();
                });
            }
        });

    }

    size(s: number) {
        return Helpers.size(s);
    }

    remove_imgs(element: Project) {
        this.service.remove_imgs(element.id).subscribe(data => {
            element.img_size = 0
        });
    }

    remove_files(element) {
        this.service.remove_files(element.id).subscribe(data => {
            element.file_size = 0
        });
    }

    add() {
        const dialogRef = this.dialog.open(ProjectAddDialogComponent, {
            width: '600px', height: '300px',
            data: {'name': ''}
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.add(result).subscribe(_ => {
                    this.change.emit();
                });

            }
        });
    }
}

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

